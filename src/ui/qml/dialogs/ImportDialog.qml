import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../theme"
import "../components"

// "Add video": file info + copy progress + video id + lane editor. Backed by `importer` and `laneEditor`.
Dialog {
    id: dialog
    modal: true
    closePolicy: Popup.NoAutoClose
    anchors.centerIn: parent
    width: parent.width - 48
    height: parent.height - 48
    padding: 20
    topPadding: 16

    background: Rectangle {
        color: Theme.bg
        radius: Theme.radius
        border.color: Theme.borderStrong
    }

    header: Item {
        implicitHeight: 56
        Label {
            anchors.left: parent.left
            anchors.leftMargin: 20
            anchors.verticalCenter: parent.verticalCenter
            text: "Add video"
            color: Theme.text
            font.pixelSize: 20
            font.weight: Font.DemiBold
        }
    }

    contentItem: ColumnLayout {
        spacing: Theme.gap

        Card {
            Layout.fillWidth: true
            padding: 14

            RowLayout {
                Layout.fillWidth: true
                spacing: 24

                ColumnLayout {
                    Layout.fillWidth: true
                    spacing: 4
                    Label {
                        Layout.fillWidth: true
                        text: importer.fileName
                        color: Theme.text
                        font.pixelSize: 15
                        font.weight: Font.DemiBold
                        elide: Text.ElideMiddle
                    }
                    Label {
                        Layout.fillWidth: true
                        text: importer.infoText
                        color: Theme.textDim
                        font.pixelSize: 12
                        elide: Text.ElideRight
                    }
                    Label {
                        text: importer.needsCopy ? `Will be copied to ${importer.targetText}` : "Already in data/raw — no copy needed"
                        color: Theme.textFaint
                        font.pixelSize: 12
                    }
                }

                ColumnLayout {
                    visible: importer.needsCopy
                    Layout.preferredWidth: 260
                    spacing: 4
                    Label {
                        text: importer.copyState === "copying" ? `Copying… ${(importer.copyProgress * 100).toFixed(0)}%`
                            : importer.copyState === "done" ? "Copied"
                            : importer.copyState === "failed" ? "Copy failed"
                            : ""
                        color: importer.copyState === "failed" ? Theme.danger
                             : importer.copyState === "done" ? Theme.success : Theme.textDim
                        font.pixelSize: 12
                    }
                    ProgressBar {
                        Layout.fillWidth: true
                        value: importer.copyProgress
                    }
                }

                ColumnLayout {
                    Layout.preferredWidth: 300
                    spacing: 4
                    Label {
                        text: "Video ID"
                        color: Theme.textDim
                        font.pixelSize: 12
                    }
                    TextField {
                        id: idField
                        Layout.fillWidth: true
                        font.family: Theme.mono
                        Binding on text {
                            value: importer.videoId
                            when: !idField.activeFocus
                            restoreMode: Binding.RestoreNone
                        }
                        onTextEdited: importer.setVideoId(text)
                    }
                    Label {
                        Layout.fillWidth: true
                        text: importer.videoIdError
                        visible: text !== ""
                        color: Theme.danger
                        font.pixelSize: 12
                        wrapMode: Text.Wrap
                    }
                }
            }
        }

        LaneEditorView {
            id: editorView
            Layout.fillWidth: true
            Layout.fillHeight: true
        }
    }

    footer: Item {
        implicitHeight: 68
        RowLayout {
            anchors.fill: parent
            anchors.leftMargin: 20
            anchors.rightMargin: 20
            spacing: 10
            Label {
                visible: importer.savePending
                text: "Saving as soon as the copy finishes…"
                color: Theme.textDim
            }
            Item { Layout.fillWidth: true }
            Button {
                text: "Cancel"
                Layout.preferredWidth: 120
                onClicked: dialog.reject()
            }
            Button {
                text: importer.savePending ? "Waiting for copy…" : "Save video"
                highlighted: true
                Layout.preferredWidth: 160
                enabled: laneEditor.canSave && importer.videoIdError === "" && importer.copyState !== "failed" && !importer.savePending
                onClicked: importer.save()
            }
        }
    }

    onAboutToShow: editorView.attachSinks()
    onRejected: importer.cancel()
}
