import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../theme"
import "../components"

Dialog {
    id: dialog
    modal: true
    closePolicy: Popup.NoAutoClose
    anchors.centerIn: parent
    width: parent.width - 48
    height: parent.height - 48
    padding: 20
    topPadding: 8

    function openFor(videoId) {
        if (laneEditor.openForEdit(videoId))
            open()
    }

    background: Rectangle {
        color: Theme.bg
        radius: Theme.radius
        border.color: Theme.borderStrong
    }

    header: Item {
        implicitHeight: 56
        RowLayout {
            anchors.fill: parent
            anchors.leftMargin: 20
            anchors.rightMargin: 20
            Label {
                text: "Edit lanes"
                color: Theme.text
                font.pixelSize: 20
                font.weight: Font.DemiBold
            }
            Label {
                text: "· " + laneEditor.videoId
                color: Theme.textDim
                font.pixelSize: 16
            }
            Item { Layout.fillWidth: true }
        }
    }

    contentItem: LaneEditorView { id: editorView }

    footer: Item {
        implicitHeight: 68
        RowLayout {
            anchors.fill: parent
            anchors.leftMargin: 20
            anchors.rightMargin: 20
            spacing: 10
            Item { Layout.fillWidth: true }
            Button {
                text: "Cancel"
                Layout.preferredWidth: 120
                onClicked: dialog.reject()
            }
            Button {
                text: "Save lanes"
                highlighted: true
                Layout.preferredWidth: 160
                enabled: laneEditor.canSave
                onClicked: if (laneEditor.save()) dialog.close()
            }
        }
    }

    onAboutToShow: editorView.attachSinks()
    onRejected: laneEditor.close()
}
