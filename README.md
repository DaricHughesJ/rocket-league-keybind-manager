# Rocket League Keybind Manager

A Windows desktop utility for creating, managing, backing up, and applying Rocket League keybind and controller-settings profiles.

## Included

- PowerShell settings manager
- One-click Windows launcher
- Profile import/export and local backups
- Rocket League input and system-settings snapshots

## Run

Double-click `START-RL-SETTINGS-CLONE-v4.bat` or run it from PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\RL-Settings-Clone-v4.ps1
```

## Requirements

- Windows 10/11
- PowerShell 5.1 or newer
- Rocket League installed locally

## Safety

The app keeps local configuration snapshots and backups before applying changes.
