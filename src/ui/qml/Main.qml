import QtQuick
import QtQuick.Controls
import QtQuick.Dialogs
import "theme"
import "pages"
import "dialogs"
import "components"

ApplicationWindow {
    id: window
    width: 1500
    height: 940
    minimumWidth: 1180
    minimumHeight: 740
    visible: true
    title: stack.currentItem === workspacePage ? `Swimmer Tracker — ${workspace.videoId}` : "Swimmer Tracker"
    color: Theme.bg

    readonly property bool dialogOpen: importDialog.visible || laneDialog.visible || clipDialog.visible

    StackView {
        id: stack
        objectName: "stack"
        anchors.fill: parent
        initialItem: libraryPage
        // instant page switches: the style's render-thread animators can stall while the window is not exposed
        pushEnter: Transition {}
        pushExit: Transition {}
        popEnter: Transition {}
        popExit: Transition {}
    }

    LibraryPage {
        id: libraryPage
        objectName: "libraryPage"
        visible: false
        onOpenVideo: (videoId) => {
            if (workspace.openVideo(videoId))
                stack.push(workspacePage)
        }
        onAddVideo: fileDialog.open()
        onEditLanes: (videoId) => laneDialog.openFor(videoId)
    }

    WorkspacePage {
        id: workspacePage
        objectName: "workspacePage"
        visible: false
        shortcutsEnabled: stack.currentItem === workspacePage && !window.dialogOpen
        onBack: {
            workspace.stop()
            workspace.closeVideo()
            stack.pop()
        }
        onEditLanes: laneDialog.openFor(workspace.videoId)
        onCreateClip: clipDialog.openFor()
    }

    FileDialog {
        id: fileDialog
        title: "Choose a video"
        currentFolder: library.rawDirUrl
        nameFilters: ["Video files (*.mp4 *.mov *.avi *.mkv *.m4v)", "All files (*)"]
        onAccepted: {
            if (importer.begin(selectedFile))
                importDialog.open()
        }
    }

    ImportDialog { id: importDialog; objectName: "importDialog" }
    LaneEditDialog { id: laneDialog; objectName: "laneDialog" }
    CreateClipDialog { id: clipDialog; objectName: "clipDialog" }
    Toast { id: toast; objectName: "toast" }

    Connections {
        target: library
        function onErrorOccurred(message) { toast.show(message, "error") }
    }
    Connections {
        target: workspace
        function onErrorOccurred(message) { toast.show(message, "error") }
        function onInfoMessage(message) { toast.show(message, "success") }
    }
    Connections {
        target: importer
        function onErrorOccurred(message) { toast.show(message, "error") }
        function onFinished(videoId) {
            importDialog.close()
            toast.show(`Video '${videoId}' added.`, "success")
        }
    }
    Connections {
        target: laneEditor
        function onErrorOccurred(message) { toast.show(message, "error") }
        function onSaved(videoId) { toast.show(`Lanes of '${videoId}' saved.`, "success") }
    }
}
