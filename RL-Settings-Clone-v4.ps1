#requires -Version 5.1
<#
RL Settings Clone v4.0
- Built-in sanitized Rocket League save-data settings profile extracted from the user's uploaded DBE_Production.
- Transplants only whitelisted settings objects into another account's encrypted .save files.
- Preserves target account progression/inventory/identity objects.
- Can also snapshot/apply local TAGame\Config files for machine-local settings.
#>

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$CoreSource = @'
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Security.Cryptography;

public sealed class RLObjectEntry
{
    public string TypeName;
    public uint ObjectIndex;
    public byte[] Segment;
}

public sealed class RLSaveContainer
{
    public int EngineVersion;
    public int LicenseeVersion;
    public int TypeVersion;
    public byte[] RootSegment;
    public List<RLObjectEntry> Objects = new List<RLObjectEntry>();

    public bool ReplaceSegment(string typeName, int ordinal, byte[] segment)
    {
        if (typeName == null || segment == null || segment.Length < 4) return false;
        if (segment[0] != 0xFF || segment[1] != 0xFF || segment[2] != 0xFF || segment[3] != 0xFF)
            throw new InvalidDataException("Profile segment does not begin with the Rocket League object header.");

        int seen = 0;
        for (int i = 0; i < Objects.Count; i++)
        {
            if (String.Equals(Objects[i].TypeName, typeName, StringComparison.Ordinal))
            {
                if (seen == ordinal)
                {
                    Objects[i].Segment = (byte[])segment.Clone();
                    return true;
                }
                seen++;
            }
        }
        return false;
    }

    public byte[] GetSegment(string typeName, int ordinal)
    {
        int seen = 0;
        for (int i = 0; i < Objects.Count; i++)
        {
            if (String.Equals(Objects[i].TypeName, typeName, StringComparison.Ordinal))
            {
                if (seen == ordinal) return (byte[])Objects[i].Segment.Clone();
                seen++;
            }
        }
        return null;
    }

    public string GetSegmentHash(string typeName, int ordinal)
    {
        byte[] data = GetSegment(typeName, ordinal);
        if (data == null) return null;
        return RLSaveCodec.Sha256(data);
    }

    public byte[] ToFileBytes()
    {
        return RLSaveCodec.Build(this);
    }
}

public static class RLSaveCodec
{
    private static readonly byte[] Key = new byte[] {
        0xD7,0x8C,0x32,0x4A,0x94,0x42,0x94,0x3C,0x6D,0x65,0xCE,0x98,0x81,0x85,0x4C,0x41,
        0x68,0x99,0x22,0x0C,0xC7,0xA1,0x46,0x40,0x93,0x9B,0x96,0x3C,0x93,0x2A,0x6F,0xAF
    };

    private const uint SaveSeed = 0xEFCBF201u;
    private const uint Poly = 0x04C11DB7u;
    private const uint Foosball = 0xF005BA11u;
    private const uint Magic = 0x7FFFFFFFu;

    public static string Sha256(byte[] data)
    {
        using (SHA256 sha = SHA256.Create())
        {
            byte[] hash = sha.ComputeHash(data);
            StringBuilder sb = new StringBuilder(hash.Length * 2);
            for (int i = 0; i < hash.Length; i++) sb.Append(hash[i].ToString("X2"));
            return sb.ToString();
        }
    }

    public static uint Crc32(byte[] data)
    {
        unchecked
        {
            uint crc = ~SaveSeed;
            for (int j = 0; j < data.Length; j++)
            {
                uint c = (uint)(((crc >> 24) & 0xFFu) ^ data[j]);
                c <<= 24;
                for (int i = 0; i < 8; i++)
                {
                    c = (c & 0x80000000u) != 0 ? (c << 1) ^ Poly : (c << 1);
                }
                crc = (crc << 8) ^ c;
            }
            return ~crc;
        }
    }

    private static byte[] TransformAes(byte[] data, bool encrypt)
    {
        using (Aes aes = Aes.Create())
        {
            aes.KeySize = 256;
            aes.BlockSize = 128;
            aes.Mode = CipherMode.ECB;
            aes.Padding = PaddingMode.None;
            aes.Key = Key;
            using (ICryptoTransform tx = encrypt ? aes.CreateEncryptor() : aes.CreateDecryptor())
            {
                return tx.TransformFinalBlock(data, 0, data.Length);
            }
        }
    }

    private static string ReadUeString(BinaryReader br)
    {
        int length = br.ReadInt32();
        if (length == 0) return null;
        if (length > 0)
        {
            byte[] bytes = br.ReadBytes(length);
            if (bytes.Length != length) throw new EndOfStreamException();
            if (bytes[length - 1] != 0) throw new InvalidDataException("UE string is not NUL terminated.");
            return Encoding.GetEncoding(1252).GetString(bytes, 0, length - 1);
        }

        int chars = checked(-length);
        byte[] raw = br.ReadBytes(checked(chars * 2));
        if (raw.Length != chars * 2) throw new EndOfStreamException();
        if (raw.Length < 2 || raw[raw.Length - 1] != 0 || raw[raw.Length - 2] != 0)
            throw new InvalidDataException("UTF-16 UE string is not NUL terminated.");
        return Encoding.Unicode.GetString(raw, 0, raw.Length - 2);
    }

    private static void WriteUeString(BinaryWriter bw, string value)
    {
        if (value == null)
        {
            bw.Write(0);
            return;
        }

        // Rocket League object type names are ASCII/ANSI.
        byte[] bytes = Encoding.GetEncoding(1252).GetBytes(value);
        bw.Write(bytes.Length + 1);
        bw.Write(bytes);
        bw.Write((byte)0);
    }

    public static RLSaveContainer Load(string path)
    {
        return Parse(File.ReadAllBytes(path));
    }

    public static RLSaveContainer Parse(byte[] file)
    {
        if (file == null || file.Length < 8) throw new InvalidDataException("Save file is too short.");

        uint encryptedLength = BitConverter.ToUInt32(file, 0);
        uint storedCrc = BitConverter.ToUInt32(file, 4);
        if (encryptedLength == 0 || (encryptedLength % 16) != 0)
            throw new InvalidDataException("Encrypted payload length is invalid.");
        if ((long)encryptedLength + 8L > file.LongLength)
            throw new InvalidDataException("Encrypted payload length exceeds file length.");

        byte[] cipher = new byte[encryptedLength];
        Buffer.BlockCopy(file, 8, cipher, 0, (int)encryptedLength);

        uint computed = Crc32(cipher);
        if (computed != storedCrc)
            throw new InvalidDataException("Rocket League save CRC check failed.");

        byte[] payload = TransformAes(cipher, false);

        using (MemoryStream ms = new MemoryStream(payload, false))
        using (BinaryReader br = new BinaryReader(ms))
        {
            uint f = br.ReadUInt32();
            uint m = br.ReadUInt32();
            if (f != Foosball || m != Magic)
                throw new InvalidDataException("Decrypted Rocket League save header is invalid.");

            RLSaveContainer result = new RLSaveContainer();
            result.EngineVersion = br.ReadInt32();
            result.LicenseeVersion = br.ReadInt32();
            result.TypeVersion = br.ReadInt32();

            int saveDataLength = br.ReadInt32();
            if (saveDataLength < 4) throw new InvalidDataException("Save-data length is invalid.");
            int blobLength = checked(saveDataLength - 4);
            byte[] blob = br.ReadBytes(blobLength);
            if (blob.Length != blobLength) throw new EndOfStreamException();

            int count = br.ReadInt32();
            if (count < 0 || count > 10000) throw new InvalidDataException("Object count is invalid.");

            string[] types = new string[count];
            uint[] positions = new uint[count];
            uint[] objectIndexes = new uint[count];

            for (int i = 0; i < count; i++)
            {
                types[i] = ReadUeString(br);
                positions[i] = br.ReadUInt32();
                objectIndexes[i] = br.ReadUInt32();
            }

            int firstStart = count > 0 ? checked((int)positions[0] - 4) : blob.Length;
            if (firstStart < 0 || firstStart > blob.Length)
                throw new InvalidDataException("First object position is invalid.");

            result.RootSegment = new byte[firstStart];
            Buffer.BlockCopy(blob, 0, result.RootSegment, 0, firstStart);

            for (int i = 0; i < count; i++)
            {
                int start = checked((int)positions[i] - 4);
                int end = i + 1 < count ? checked((int)positions[i + 1] - 4) : blob.Length;
                if (start < 0 || end < start || end > blob.Length)
                    throw new InvalidDataException("Object positions are invalid.");

                byte[] segment = new byte[end - start];
                Buffer.BlockCopy(blob, start, segment, 0, segment.Length);
                if (segment.Length < 4 ||
                    segment[0] != 0xFF || segment[1] != 0xFF ||
                    segment[2] != 0xFF || segment[3] != 0xFF)
                    throw new InvalidDataException("Object header is invalid.");

                result.Objects.Add(new RLObjectEntry {
                    TypeName = types[i],
                    ObjectIndex = objectIndexes[i],
                    Segment = segment
                });
            }

            return result;
        }
    }

    public static byte[] Build(RLSaveContainer c)
    {
        if (c == null) throw new ArgumentNullException("c");
        if (c.RootSegment == null) throw new InvalidDataException("Root segment is missing.");

        byte[] payload;
        using (MemoryStream pms = new MemoryStream())
        using (BinaryWriter bw = new BinaryWriter(pms))
        {
            bw.Write(Foosball);
            bw.Write(Magic);
            bw.Write(c.EngineVersion);
            bw.Write(c.LicenseeVersion);
            bw.Write(c.TypeVersion);

            long lengthFieldOffset = pms.Position;
            bw.Write(0); // saveDataLength patched later

            MemoryStream blobStream = new MemoryStream();
            blobStream.Write(c.RootSegment, 0, c.RootSegment.Length);
            uint[] positions = new uint[c.Objects.Count];

            for (int i = 0; i < c.Objects.Count; i++)
            {
                positions[i] = checked((uint)blobStream.Length + 4u);
                byte[] seg = c.Objects[i].Segment;
                if (seg == null || seg.Length < 4)
                    throw new InvalidDataException("Object segment is missing.");
                blobStream.Write(seg, 0, seg.Length);
            }

            byte[] blob = blobStream.ToArray();
            int saveDataLength = checked(blob.Length + 4);
            bw.Write(blob);

            bw.Write(c.Objects.Count);
            for (int i = 0; i < c.Objects.Count; i++)
            {
                WriteUeString(bw, c.Objects[i].TypeName);
                bw.Write(positions[i]);
                bw.Write(c.Objects[i].ObjectIndex);
            }

            long finalUnpadded = pms.Length;
            pms.Position = lengthFieldOffset;
            bw.Write(saveDataLength);
            pms.Position = finalUnpadded;

            int pad = (int)((16 - (pms.Length % 16)) % 16);
            for (int i = 0; i < pad; i++) bw.Write((byte)0);
            bw.Flush();
            payload = pms.ToArray();
        }

        byte[] cipher = TransformAes(payload, true);
        uint crc = Crc32(cipher);

        using (MemoryStream outMs = new MemoryStream())
        using (BinaryWriter outBw = new BinaryWriter(outMs))
        {
            outBw.Write((uint)cipher.Length);
            outBw.Write(crc);
            outBw.Write(cipher);
            outBw.Flush();
            return outMs.ToArray();
        }
    }
}
'@

try {
    Add-Type -TypeDefinition $CoreSource -Language CSharp -ErrorAction Stop
} catch {
    [System.Windows.Forms.MessageBox]::Show(
        "Could not initialize the Rocket League save codec.`r`n`r`n$($_.Exception.Message)",
        "RL Settings Clone", "OK", "Error"
    ) | Out-Null
    exit 1
}

[System.Windows.Forms.Application]::EnableVisualStyles()

$AppName = "RL Settings Clone"
$Version = "4.0.0"
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProfilePath = Join-Path $ScriptRoot "Daric-v4.profile.json"
$DataRoot = Join-Path $ScriptRoot "RLSettingsCloneData"
$BackupRoot = Join-Path $DataRoot "Backups"
$ReportRoot = Join-Path $DataRoot "Reports"
$LogRoot = Join-Path $DataRoot "Logs"
$LocalSnapshot = Join-Path $DataRoot "LocalConfigSnapshot"

New-Item -ItemType Directory -Force -Path $DataRoot,$BackupRoot,$ReportRoot,$LogRoot | Out-Null

if (-not (Test-Path $ProfilePath)) {
    [System.Windows.Forms.MessageBox]::Show("Built-in profile is missing.",$AppName,"OK","Error") | Out-Null
    exit 1
}

$Profile = Get-Content -LiteralPath $ProfilePath -Raw -Encoding UTF8 | ConvertFrom-Json
$ProfileSettings = @($Profile.settings)
$ProfileSegments = @($Profile.segments)

if ($ProfileSettings.Count -ne 94) {
    [System.Windows.Forms.MessageBox]::Show(
        "Profile integrity error: expected 94 visible settings, loaded $($ProfileSettings.Count).",
        $AppName,"OK","Error"
    ) | Out-Null
    exit 1
}
if ($ProfileSegments.Count -lt 10) {
    [System.Windows.Forms.MessageBox]::Show("Profile segment set is incomplete.",$AppName,"OK","Error") | Out-Null
    exit 1
}

function Write-Log {
    param([string]$Message,[string]$Level="INFO")
    $file = Join-Path $LogRoot ("RLSettingsClone_" + (Get-Date -Format "yyyyMMdd") + ".log")
    Add-Content -LiteralPath $file -Encoding UTF8 -Value ("[{0}][{1}] {2}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"),$Level,$Message)
}

function Get-DocumentsPath {
    try {
        $r = Get-ItemProperty "HKCU:\Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders" -ErrorAction Stop
        if ($r.Personal) {
            $p = [Environment]::ExpandEnvironmentVariables([string]$r.Personal)
            if (Test-Path $p) { return $p }
        }
    } catch {}
    return [Environment]::GetFolderPath("MyDocuments")
}

function Get-RLTagamePath {
    Join-Path (Get-DocumentsPath) "My Games\Rocket League\TAGame"
}

function Get-DefaultSaveFolder {
    $base = Get-RLTagamePath
    $epic = Join-Path $base "SaveDataEpic\DBE_Production"
    $legacy = Join-Path $base "SaveData\DBE_Production"
    if (Test-Path $epic) { return $epic }
    if (Test-Path $legacy) { return $legacy }
    return $epic
}

function Get-ConfigPath {
    Join-Path (Get-RLTagamePath) "Config"
}

function Assert-RLClosed {
    $p = Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.ProcessName -like "RocketLeague*" }
    if ($p) { throw "Rocket League is running. Close Rocket League before changing save/config files." }
}

function Get-SegmentCategories {
    param($Segment)
    return @($Segment.categories)
}

function Category-Selected {
    param([string]$Category)
    switch ($Category) {
        "Gameplay" { return $chkGameplay.Checked }
        "Camera" { return $chkCamera.Checked }
        "Controls" { return $chkControls.Checked }
        "Interface" { return $chkInterface.Checked }
        "Video" { return $chkVideo.Checked }
        "Audio" { return $chkAudio.Checked }
        "Chat" { return $chkChat.Checked }
        default { return $false }
    }
}

function Segment-IsSelected {
    param($Segment)
    foreach ($cat in @(Get-SegmentCategories $Segment)) {
        if (Category-Selected ([string]$cat)) { return $true }
    }
    return $false
}

function Get-AccountGroups {
    param([string]$Folder)
    $groups = @{}
    if (-not (Test-Path $Folder)) { return @() }

    foreach ($f in Get-ChildItem -LiteralPath $Folder -Filter "*.save" -File -ErrorAction SilentlyContinue) {
        $m = [regex]::Match($f.Name,'^(?<id>[0-9A-Fa-f]{32})(?:_(?:1|2))?\.save$')
        if (-not $m.Success) { continue }
        $id = $m.Groups["id"].Value.ToLowerInvariant()
        if (-not $groups.ContainsKey($id)) {
            $groups[$id] = New-Object System.Collections.ArrayList
        }
        [void]$groups[$id].Add($f)
    }

    $result = @()
    foreach ($id in $groups.Keys) {
        $files = @($groups[$id] | Sort-Object LastWriteTime -Descending)
        $latest = $files | Select-Object -First 1
        $result += [pscustomobject]@{
            Id = $id
            Files = $files
            Latest = $latest.LastWriteTime
            LatestFile = $latest.FullName
        }
    }
    return @($result | Sort-Object Latest -Descending)
}

function Get-ProfileMatchCount {
    param([string]$Path)
    try {
        $c = [RLSaveCodec]::Load($Path)
        $match = 0
        $possible = 0
        foreach ($seg in $ProfileSegments) {
            $possible++
            $actual = $c.GetSegmentHash([string]$seg.type,[int]$seg.ordinal)
            if ($actual -and $actual -eq [string]$seg.sha256) { $match++ }
        }
        return [pscustomobject]@{ Match=$match; Possible=$possible }
    } catch {
        return [pscustomobject]@{ Match=-1; Possible=$ProfileSegments.Count }
    }
}

function Copy-ConfigTree {
    param([string]$Source,[string]$Destination)
    if (-not (Test-Path $Source)) { return 0 }
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    $count=0
    foreach ($f in Get-ChildItem -LiteralPath $Source -File -Recurse -ErrorAction SilentlyContinue |
             Where-Object { $_.Extension.ToLowerInvariant() -in @(".ini",".cfg") }) {
        $rel = $f.FullName.Substring($Source.TrimEnd('\').Length).TrimStart('\')
        $dest = Join-Path $Destination $rel
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dest) | Out-Null
        Copy-Item -LiteralPath $f.FullName -Destination $dest -Force
        $count++
    }
    return $count
}

function Capture-LocalConfig {
    Assert-RLClosed
    $config = Get-ConfigPath
    if (-not (Test-Path $config)) { throw "Rocket League Config folder not found: $config" }
    if (Test-Path $LocalSnapshot) { Remove-Item -LiteralPath $LocalSnapshot -Recurse -Force }
    $count = Copy-ConfigTree $config $LocalSnapshot
    if ($count -eq 0) { throw "No .ini/.cfg files were found to capture." }
    Write-Log "Captured $count local config files to $LocalSnapshot"
    return $count
}

function Backup-Target {
    param($Group,[bool]$IncludeConfig)
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $dir = Join-Path $BackupRoot ("$stamp`_$($Group.Id)")
    $saveDir = Join-Path $dir "SaveFiles"
    New-Item -ItemType Directory -Force -Path $saveDir | Out-Null

    foreach ($f in @($Group.Files)) {
        Copy-Item -LiteralPath $f.FullName -Destination (Join-Path $saveDir $f.Name) -Force
    }

    if ($IncludeConfig) {
        $config=Get-ConfigPath
        if (Test-Path $config) {
            [void](Copy-ConfigTree $config (Join-Path $dir "Config"))
        }
    }

    [pscustomobject]@{
        Created=(Get-Date).ToString("o")
        SaveFolder=$txtSaveFolder.Text
        ConfigPath=(Get-ConfigPath)
        GroupId=$Group.Id
    } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $dir "backup.json") -Encoding UTF8

    return $dir
}

function Apply-SaveProfileToFile {
    param([string]$Path)

    $container = [RLSaveCodec]::Load($Path)
    $expected = New-Object System.Collections.ArrayList
    $missing = New-Object System.Collections.ArrayList

    foreach ($seg in $ProfileSegments) {
        if (-not (Segment-IsSelected $seg)) { continue }
        $bytes = [Convert]::FromBase64String([string]$seg.data)
        $ok = $container.ReplaceSegment([string]$seg.type,[int]$seg.ordinal,$bytes)
        if ($ok) {
            [void]$expected.Add([pscustomobject]@{
                Type=[string]$seg.type
                Ordinal=[int]$seg.ordinal
                Sha=[string]$seg.sha256
            })
        } else {
            [void]$missing.Add("$($seg.type) #$($seg.ordinal)")
        }
    }

    if ($expected.Count -eq 0) { throw "No selected save-data setting objects could be applied." }

    $tmp = "$Path.rlsclone.tmp"
    [IO.File]::WriteAllBytes($tmp,$container.ToFileBytes())

    # Full envelope/CRC/decrypt parse + segment verification before replacing the original.
    $verify = [RLSaveCodec]::Load($tmp)
    foreach ($e in @($expected)) {
        $actual = $verify.GetSegmentHash([string]$e.Type,[int]$e.Ordinal)
        if (-not $actual -or $actual -ne [string]$e.Sha) {
            Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue
            throw "Verification failed for $($e.Type) #$($e.Ordinal). Original file was not replaced."
        }
    }

    Move-Item -LiteralPath $tmp -Destination $Path -Force

    return [pscustomobject]@{
        Applied=$expected.Count
        Missing=@($missing)
    }
}

function Apply-LocalSnapshot {
    if (-not (Test-Path $LocalSnapshot)) { throw "No local config snapshot exists. Use CAPTURE LOCAL CONFIG first, or uncheck Apply local config." }
    $config=Get-ConfigPath
    New-Item -ItemType Directory -Force -Path $config | Out-Null
    return Copy-ConfigTree $LocalSnapshot $config
}

function Restore-Backup {
    param([string]$BackupDir)
    Assert-RLClosed
    if (-not (Test-Path $BackupDir)) { throw "Backup folder not found." }
    $metaPath=Join-Path $BackupDir "backup.json"
    if (-not (Test-Path $metaPath)) { throw "Backup metadata not found." }
    $meta=Get-Content -LiteralPath $metaPath -Raw | ConvertFrom-Json

    $saveDir=Join-Path $BackupDir "SaveFiles"
    if (Test-Path $saveDir) {
        foreach ($f in Get-ChildItem -LiteralPath $saveDir -File) {
            Copy-Item -LiteralPath $f.FullName -Destination (Join-Path ([string]$meta.SaveFolder) $f.Name) -Force
        }
    }

    $cfg=Join-Path $BackupDir "Config"
    if (Test-Path $cfg) {
        [void](Copy-ConfigTree $cfg ([string]$meta.ConfigPath))
    }
}

# ---------------- GUI ----------------

$form = New-Object Windows.Forms.Form
$form.Text = "$AppName $Version"
$form.StartPosition = "CenterScreen"
$form.Size = New-Object Drawing.Size(1240,820)
$form.MinimumSize = New-Object Drawing.Size(1060,720)
$form.Font = New-Object Drawing.Font("Segoe UI",9)
$form.BackColor = [Drawing.Color]::FromArgb(248,248,248)

$title = New-Object Windows.Forms.Label
$title.Text = "Rocket League Settings Clone v4"
$title.Font = New-Object Drawing.Font("Segoe UI Semibold",18)
$title.AutoSize=$true
$title.Location=New-Object Drawing.Point(18,12)
$form.Controls.Add($title)

$subtitle=New-Object Windows.Forms.Label
$subtitle.Text="Real DBE save-data transplant + local config snapshot. Target progression/inventory objects are left untouched."
$subtitle.AutoSize=$true
$subtitle.ForeColor=[Drawing.Color]::DimGray
$subtitle.Location=New-Object Drawing.Point(20,49)
$form.Controls.Add($subtitle)

$lblSave=New-Object Windows.Forms.Label
$lblSave.Text="DBE_Production folder"
$lblSave.AutoSize=$true
$lblSave.Location=New-Object Drawing.Point(20,80)
$form.Controls.Add($lblSave)

$txtSaveFolder=New-Object Windows.Forms.TextBox
$txtSaveFolder.Location=New-Object Drawing.Point(20,101)
$txtSaveFolder.Size=New-Object Drawing.Size(820,25)
$txtSaveFolder.Text=Get-DefaultSaveFolder
$form.Controls.Add($txtSaveFolder)

$btnBrowse=New-Object Windows.Forms.Button
$btnBrowse.Text="Browse"
$btnBrowse.Location=New-Object Drawing.Point(850,99)
$btnBrowse.Size=New-Object Drawing.Size(80,28)
$form.Controls.Add($btnBrowse)

$btnScan=New-Object Windows.Forms.Button
$btnScan.Text="SCAN ACCOUNTS"
$btnScan.Location=New-Object Drawing.Point(940,99)
$btnScan.Size=New-Object Drawing.Size(125,28)
$form.Controls.Add($btnScan)

$btnOpen=New-Object Windows.Forms.Button
$btnOpen.Text="Open Data"
$btnOpen.Location=New-Object Drawing.Point(1075,99)
$btnOpen.Size=New-Object Drawing.Size(115,28)
$form.Controls.Add($btnOpen)

$lblAccounts=New-Object Windows.Forms.Label
$lblAccounts.Text="Detected account save groups"
$lblAccounts.AutoSize=$true
$lblAccounts.Location=New-Object Drawing.Point(20,142)
$form.Controls.Add($lblAccounts)

$accounts=New-Object Windows.Forms.DataGridView
$accounts.Location=New-Object Drawing.Point(20,163)
$accounts.Size=New-Object Drawing.Size(1170,150)
$accounts.AllowUserToAddRows=$false
$accounts.AllowUserToDeleteRows=$false
$accounts.ReadOnly=$true
$accounts.RowHeadersVisible=$false
$accounts.SelectionMode="FullRowSelect"
$accounts.MultiSelect=$false
$accounts.AutoGenerateColumns=$false
$accounts.BackgroundColor=[Drawing.Color]::White
$accounts.BorderStyle="FixedSingle"

foreach($spec in @(
    @("Account","Account save ID",340),
    @("Files","Files",70),
    @("Modified","Last modified",190),
    @("Match","Built-in profile match",220),
    @("Hint","Hint",310)
)) {
    $c=New-Object Windows.Forms.DataGridViewTextBoxColumn
    $c.Name=$spec[0];$c.HeaderText=$spec[1];$c.Width=[int]$spec[2]
    [void]$accounts.Columns.Add($c)
}
$form.Controls.Add($accounts)

$lblCats=New-Object Windows.Forms.Label
$lblCats.Text="Apply categories"
$lblCats.AutoSize=$true
$lblCats.Location=New-Object Drawing.Point(20,328)
$form.Controls.Add($lblCats)

function New-CatCheck([string]$Text,[int]$X) {
    $cb=New-Object Windows.Forms.CheckBox
    $cb.Text=$Text
    $cb.Checked=$true
    $cb.AutoSize=$true
    $cb.Location=New-Object Drawing.Point($X,349)
    $form.Controls.Add($cb)
    return $cb
}
$chkGameplay=New-CatCheck "Gameplay" 20
$chkCamera=New-CatCheck "Camera" 115
$chkControls=New-CatCheck "Controls + bindings" 200
$chkInterface=New-CatCheck "Interface" 345
$chkVideo=New-CatCheck "Video" 435
$chkAudio=New-CatCheck "Audio" 505
$chkChat=New-CatCheck "Chat / quick chat" 575

$chkLocal=New-Object Windows.Forms.CheckBox
$chkLocal.Text="Apply captured local Config (.ini/.cfg)"
$chkLocal.Checked=$true
$chkLocal.AutoSize=$true
$chkLocal.Location=New-Object Drawing.Point(720,349)
$form.Controls.Add($chkLocal)

$btnCapture=New-Object Windows.Forms.Button
$btnCapture.Text="CAPTURE LOCAL CONFIG"
$btnCapture.Location=New-Object Drawing.Point(970,341)
$btnCapture.Size=New-Object Drawing.Size(220,34)
$form.Controls.Add($btnCapture)

$lblSettings=New-Object Windows.Forms.Label
$lblSettings.Text="Your 94 screenshot settings"
$lblSettings.AutoSize=$true
$lblSettings.Location=New-Object Drawing.Point(20,388)
$form.Controls.Add($lblSettings)

$settingsGrid=New-Object Windows.Forms.DataGridView
$settingsGrid.Location=New-Object Drawing.Point(20,409)
$settingsGrid.Size=New-Object Drawing.Size(1170,250)
$settingsGrid.Anchor="Top,Left,Right,Bottom"
$settingsGrid.AllowUserToAddRows=$false
$settingsGrid.AllowUserToDeleteRows=$false
$settingsGrid.ReadOnly=$true
$settingsGrid.RowHeadersVisible=$false
$settingsGrid.SelectionMode="FullRowSelect"
$settingsGrid.AutoGenerateColumns=$false
$settingsGrid.BackgroundColor=[Drawing.Color]::White
foreach($spec in @(
    @("Category","Category",110),
    @("Setting","Setting",310),
    @("Value","Your value",310),
    @("Coverage","How v4 applies it",400)
)) {
    $c=New-Object Windows.Forms.DataGridViewTextBoxColumn
    $c.Name=$spec[0];$c.HeaderText=$spec[1];$c.Width=[int]$spec[2]
    [void]$settingsGrid.Columns.Add($c)
}
$form.Controls.Add($settingsGrid)

$btnApply=New-Object Windows.Forms.Button
$btnApply.Text="APPLY TO SELECTED ACCOUNT"
$btnApply.Location=New-Object Drawing.Point(20,678)
$btnApply.Size=New-Object Drawing.Size(250,40)
$btnApply.Anchor="Left,Bottom"
$form.Controls.Add($btnApply)

$btnRestore=New-Object Windows.Forms.Button
$btnRestore.Text="RESTORE LAST BACKUP"
$btnRestore.Location=New-Object Drawing.Point(280,678)
$btnRestore.Size=New-Object Drawing.Size(200,40)
$btnRestore.Anchor="Left,Bottom"
$form.Controls.Add($btnRestore)

$lblLocal=New-Object Windows.Forms.Label
$lblLocal.AutoSize=$true
$lblLocal.Location=New-Object Drawing.Point(500,690)
$lblLocal.Anchor="Left,Bottom"
$form.Controls.Add($lblLocal)

$status=New-Object Windows.Forms.TextBox
$status.Location=New-Object Drawing.Point(20,729)
$status.Size=New-Object Drawing.Size(1170,48)
$status.Multiline=$true
$status.ReadOnly=$true
$status.ScrollBars="Vertical"
$status.Anchor="Left,Right,Bottom"
$status.BackColor=[Drawing.Color]::White
$form.Controls.Add($status)

$script:Groups=@()
$script:LastBackup=$null

function Update-LocalLabel {
    if (Test-Path $LocalSnapshot) {
        $count=@(Get-ChildItem -LiteralPath $LocalSnapshot -File -Recurse -ErrorAction SilentlyContinue).Count
        $lblLocal.Text="Local snapshot: $count files"
    } else {
        $lblLocal.Text="Local snapshot: NOT CAPTURED"
    }
}

function Load-SettingsGrid {
    $settingsGrid.Rows.Clear()
    foreach($s in $ProfileSettings) {
        $i=$settingsGrid.Rows.Add()
        $r=$settingsGrid.Rows[$i]
        $r.Cells["Category"].Value=[string]$s.category
        $r.Cells["Setting"].Value=[string]$s.name
        $r.Cells["Value"].Value=[string]$s.value
        $r.Cells["Coverage"].Value=[string]$s.coverage
    }
}

function Scan-Accounts {
    $accounts.Rows.Clear()
    $script:Groups=@(Get-AccountGroups $txtSaveFolder.Text)
    if ($script:Groups.Count -eq 0) {
        $status.Text="No Rocket League .save account groups found in: $($txtSaveFolder.Text)"
        return
    }

    $rank=0
    foreach($g in $script:Groups) {
        $rank++
        $m=Get-ProfileMatchCount $g.LatestFile
        $hint=""
        if($m.Match -eq $m.Possible) { $hint="Matches your uploaded master settings" }
        elseif($rank -eq 1) { $hint="Most recently modified account" }

        $i=$accounts.Rows.Add()
        $r=$accounts.Rows[$i]
        $r.Tag=$g
        $r.Cells["Account"].Value=$g.Id
        $r.Cells["Files"].Value=@($g.Files).Count
        $r.Cells["Modified"].Value=$g.Latest.ToString("yyyy-MM-dd HH:mm:ss")
        $r.Cells["Match"].Value=if($m.Match -ge 0){"$($m.Match) / $($m.Possible) objects"}else{"Unreadable"}
        $r.Cells["Hint"].Value=$hint
    }
    if($accounts.Rows.Count -gt 0){$accounts.Rows[0].Selected=$true}
    $status.Text="Found $($script:Groups.Count) account save groups. Select the account you want to receive the preset."
}

$btnBrowse.Add_Click({
    $d=New-Object Windows.Forms.FolderBrowserDialog
    $d.Description="Select Rocket League DBE_Production folder"
    $d.SelectedPath=$txtSaveFolder.Text
    if($d.ShowDialog() -eq "OK"){$txtSaveFolder.Text=$d.SelectedPath;Scan-Accounts}
})
$btnScan.Add_Click({try{Scan-Accounts}catch{$status.Text="ERROR: $($_.Exception.Message)";Write-Log ($_|Out-String) "ERROR"}})
$btnOpen.Add_Click({Start-Process explorer.exe $DataRoot})

$btnCapture.Add_Click({
    try{
        $count=Capture-LocalConfig
        Update-LocalLabel
        $status.Text="Captured $count local Rocket League config files. These are applied with the save profile when enabled."
    }catch{
        Write-Log ($_|Out-String) "ERROR"
        $status.Text="ERROR: $($_.Exception.Message)"
        [Windows.Forms.MessageBox]::Show($_.Exception.Message,$AppName,"OK","Error")|Out-Null
    }
})

$btnApply.Add_Click({
    try{
        Assert-RLClosed
        if($accounts.SelectedRows.Count -lt 1){throw "Select an account save group first."}
        $group=$accounts.SelectedRows[0].Tag
        if($null -eq $group){throw "Selected account row is invalid."}

        $selectedCats=@()
        foreach($cat in @("Gameplay","Camera","Controls","Interface","Video","Audio","Chat")){
            if(Category-Selected $cat){$selectedCats+=$cat}
        }
        if($selectedCats.Count -eq 0 -and -not $chkLocal.Checked){throw "Select at least one settings category or local config."}

        $match=Get-ProfileMatchCount $group.LatestFile
        $warning="Apply your captured settings to account save ID:`r`n$($group.Id)`r`n`r`n"
        if($match.Match -eq $match.Possible){
            $warning+="This account already matches the built-in master save profile.`r`n`r`n"
        }
        $warning+="A complete backup is created first. Only whitelisted settings objects are transplanted; progression, inventory, rank, XP, loadouts, and account identity objects are not copied."
        $answer=[Windows.Forms.MessageBox]::Show($warning,$AppName,"YesNo","Warning")
        if($answer -ne "Yes"){return}

        $backup=Backup-Target $group $chkLocal.Checked
        $script:LastBackup=$backup

        $reportLines=New-Object System.Collections.ArrayList
        [void]$reportLines.Add("RL Settings Clone v4 apply report")
        [void]$reportLines.Add("Time: $(Get-Date -Format o)")
        [void]$reportLines.Add("Target group: $($group.Id)")
        [void]$reportLines.Add("Categories: $($selectedCats -join ', ')")
        [void]$reportLines.Add("Backup: $backup")
        [void]$reportLines.Add("")

        $totalApplied=0
        foreach($f in @($group.Files)){
            $r=Apply-SaveProfileToFile $f.FullName
            $totalApplied += $r.Applied
            [void]$reportLines.Add("$($f.Name): applied $($r.Applied) settings objects")
            if(@($r.Missing).Count -gt 0){
                [void]$reportLines.Add("  Missing: $(@($r.Missing) -join ', ')")
            }
        }

        $configCount=0
        if($chkLocal.Checked){
            $configCount=Apply-LocalSnapshot
            [void]$reportLines.Add("Local Config files applied: $configCount")
        }

        $reportPath=Join-Path $ReportRoot ("Apply_"+(Get-Date -Format "yyyyMMdd_HHmmss")+"_"+$group.Id+".txt")
        $reportLines | Set-Content -LiteralPath $reportPath -Encoding UTF8

        Scan-Accounts
        $status.Text="SUCCESS: applied $totalApplied save-setting objects across $(@($group.Files).Count) save file(s), plus $configCount local config file(s). Backup: $backup"
        [Windows.Forms.MessageBox]::Show(
            "Settings applied and cryptographically verified.`r`n`r`nSave objects: $totalApplied`r`nConfig files: $configCount`r`n`r`nReport: $reportPath",
            $AppName,"OK","Information")|Out-Null
    }catch{
        $detail=$_|Out-String
        Write-Log $detail "ERROR"
        $status.Text="ERROR: $($_.Exception.Message)"
        [Windows.Forms.MessageBox]::Show(
            "$($_.Exception.Message)`r`n`r`nThe original files remain available in the backup if the backup step completed.",
            $AppName,"OK","Error")|Out-Null
    }
})

$btnRestore.Add_Click({
    try{
        if(-not $script:LastBackup){throw "No backup has been created during this run."}
        $answer=[Windows.Forms.MessageBox]::Show("Restore the last backup?`r`n$($script:LastBackup)",$AppName,"YesNo","Warning")
        if($answer -ne "Yes"){return}
        Restore-Backup $script:LastBackup
        Scan-Accounts
        $status.Text="Restored: $($script:LastBackup)"
    }catch{
        Write-Log ($_|Out-String) "ERROR"
        $status.Text="ERROR: $($_.Exception.Message)"
    }
})

Load-SettingsGrid
Update-LocalLabel
try{Scan-Accounts}catch{$status.Text="Initial scan error: $($_.Exception.Message)";Write-Log ($_|Out-String) "ERROR"}
Write-Log "$AppName $Version started."
[void]$form.ShowDialog()
