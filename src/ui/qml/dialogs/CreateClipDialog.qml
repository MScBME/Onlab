import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../theme"

// Saves the currently viewed range as a clip in data/clips.json; only the description is user input.
Dialog {
    id: dialog
    modal: true
    anchors.centerIn: parent
    width: 520
    padding: 20
    property var proposal: ({})
    property string errorText: ""

    function openFor() {
        proposal = workspace.proposeClip()
        errorText = proposal.error || ""
        descriptionField.text = ""
        open()
        descriptionField.forceActiveFocus()
    }

    function save() {
        const error = workspace.createClip(descriptionField.text)
        if (error)
            errorText = error
        else
            close()
    }

    background: Rectangle {
        color: Theme.surface
        radius: Theme.radius
        border.color: Theme.borderStrong
    }

    header: Item {
        implicitHeight: 52
        Label {
            anchors.left: parent.left
            anchors.leftMargin: 20
            anchors.bottom: parent.bottom
            text: "Create clip"
            color: Theme.text
            font.pixelSize: 18
            font.weight: Font.DemiBold
        }
    }

    contentItem: ColumnLayout {
        spacing: 14

        GridLayout {
            Layout.fillWidth: true
            visible: !dialog.proposal.error
            columns: 2
            columnSpacing: 18
            rowSpacing: 6
            Repeater {
                model: [
                    ["Clip ID", dialog.proposal.id || ""],
                    ["Video", dialog.proposal.videoId || ""],
                    ["Lane", String(dialog.proposal.lane || "")],
                    ["Start", dialog.proposal.start || ""],
                    ["End", dialog.proposal.end || ""],
                    ["Duration", dialog.proposal.duration !== undefined ? dialog.proposal.duration.toFixed(2) + " s" : ""]
                ]
                delegate: RowLayout {
                    required property var modelData
                    Layout.columnSpan: 2
                    spacing: 18
                    Label {
                        Layout.preferredWidth: 80
                        text: modelData[0]
                        color: Theme.textDim
                    }
                    Label {
                        text: modelData[1]
                        color: Theme.text
                        font.family: Theme.mono
                    }
                }
            }
        }

        ColumnLayout {
            Layout.fillWidth: true
            spacing: 4
            Label {
                text: "Description (optional)"
                color: Theme.textDim
                font.pixelSize: 12
            }
            TextField {
                id: descriptionField
                Layout.fillWidth: true
                placeholderText: "e.g. lane 3, left, blue cap"
                onAccepted: if (saveButton.enabled) dialog.save()
            }
        }

        Label {
            Layout.fillWidth: true
            visible: !!dialog.proposal.duplicate
            text: `This range is already saved as '${dialog.proposal.duplicate}'.`
            color: Theme.warning
            wrapMode: Text.Wrap
        }
        Label {
            Layout.fillWidth: true
            visible: dialog.errorText !== ""
            text: dialog.errorText
            color: Theme.danger
            wrapMode: Text.Wrap
        }
    }

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
                id: saveButton
                text: "Save clip"
                highlighted: true
                Layout.preferredWidth: 140
                enabled: !dialog.proposal.error && !dialog.proposal.duplicate
                onClicked: dialog.save()
            }
        }
    }
}
