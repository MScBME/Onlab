; Inno Setup script of the Windows installer. It installs per user, without admin rights or a UAC prompt.
; packaging/build.py builds it and passes the version of pyproject.toml as /DAppVersion.
; Install and uninstall never touch the user's library in Documents\Swimmer Tracker or the settings.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

[Setup]
; Never change the AppId, upgrades and the uninstaller find the installed app by it.
AppId={{D2352B55-C065-4D6D-AFC6-F20A54714F36}
AppName=Swimmer Tracker
AppVersion={#AppVersion}
AppVerName=Swimmer Tracker {#AppVersion}
PrivilegesRequired=lowest
DefaultDirName={autopf}\Swimmer Tracker
DisableProgramGroupPage=yes
DisableDirPage=auto
OutputDir=dist
OutputBaseFilename=SwimmerTracker-{#AppVersion}-setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\SwimmerTracker.exe
UninstallDisplayName=Swimmer Tracker

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[InstallDelete]
; an upgrade replaces the whole bundle, so no file of an older version is left behind
Type: filesandordirs; Name: "{app}\_internal"

[Files]
Source: "dist\SwimmerTracker\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\Swimmer Tracker"; Filename: "{app}\SwimmerTracker.exe"
Name: "{autodesktop}\Swimmer Tracker"; Filename: "{app}\SwimmerTracker.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\SwimmerTracker.exe"; Description: "{cm:LaunchProgram,Swimmer Tracker}"; Flags: nowait postinstall skipifsilent
