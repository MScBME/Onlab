import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../theme"

Popup {
    id: root
    property string message: ""
    property string kind: "info"  // info | success | error
    readonly property color kindColor: kind === "error" ? Theme.danger : kind === "success" ? Theme.success : Theme.accent

    function show(text, type) {
        message = text
        kind = type || "info"
        open()
        hideTimer.restart()
    }

    x: (parent.width - width) / 2
    y: parent.height - height - 28
    width: Math.min(640, Math.max(320, content.implicitWidth + 40))
    padding: 0
    closePolicy: Popup.CloseOnPressOutside
    modal: false

    background: Rectangle {
        color: Theme.surfaceRaised
        radius: Theme.radius
        border.color: Theme.withAlpha(root.kindColor, 0.6)
        Rectangle {
            width: 4
            height: parent.height - 16
            anchors.verticalCenter: parent.verticalCenter
            x: 8
            radius: 2
            color: root.kindColor
        }
    }

    contentItem: RowLayout {
        id: content
        spacing: 12
        Label {
            Layout.fillWidth: true
            Layout.leftMargin: 24
            Layout.topMargin: 12
            Layout.bottomMargin: 12
            text: root.message
            color: Theme.text
            wrapMode: Text.Wrap
        }
        ToolButton {
            text: "✕"
            Layout.rightMargin: 6
            onClicked: root.close()
        }
    }

    Timer {
        id: hideTimer
        interval: root.kind === "error" ? 7000 : 3500
        onTriggered: root.close()
    }
}
