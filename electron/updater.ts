import { autoUpdater, UpdateDownloadedEvent } from 'electron-updater';
import { logger } from './logger';
import { preDownloadPythonForUpdate } from './python-setup';
import { getMainWindow } from './window';

export type UpdateChannel = 'latest' | 'beta' | 'alpha'

export function initAutoUpdater(
  channel: UpdateChannel = 'latest'
): void {
  if (channel !== 'latest') {
    autoUpdater.channel = channel
    autoUpdater.allowPrerelease = true
  }

  // On Windows, don't auto-install — we need to pre-download python-embed first.
  // On macOS and Linux, python is bundled in the app so auto-install is fine.
  if (process.platform === 'win32') {
    autoUpdater.autoInstallOnAppQuit = false
  }

  autoUpdater.on('update-downloaded', async (info: UpdateDownloadedEvent) => {
    if (process.platform !== 'win32') {
      // macOS / Linux: python is bundled, just install normally
      autoUpdater.quitAndInstall(false, true)
      return
    }

    // Windows: pre-download python-embed if deps changed before restarting
    const newVersion = info.version
    logger.info( `[updater] Update downloaded: v${newVersion}, checking python deps...`)

    try {
      const didDownload = await preDownloadPythonForUpdate(newVersion, (progress) => {
        // Forward progress to renderer so it can show a "Preparing update..." UI
        getMainWindow()?.webContents.send('python-update-progress', progress)
      })

      if (didDownload) {
        logger.info( '[updater] Python pre-download complete, installing update...')
      } else {
        logger.info( '[updater] No python changes needed, installing update...')
      }
    } catch (err) {
      // Pre-download failed — install anyway; the app will download at next launch
      logger.error( `[updater] Python pre-download failed, proceeding with update: ${err}`)
    }

    autoUpdater.quitAndInstall(false, true)
  })

  const update = () => {
    logger.info( 'Checking for update...');
    autoUpdater.checkForUpdatesAndNotify().catch((e) => {
      logger.error( `Failed checking for updates: ${e}`);
    });
  }

  // Check after startup, then periodically
  setTimeout(update, 5_000);
  setInterval(update, 4 * 60 * 60 * 1000);
}
