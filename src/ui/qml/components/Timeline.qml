import QtQuick
import QtQuick.Controls
import "../theme"

// Video timeline: playhead (scrubbing) plus a draggable analysis range.
// `zoomed` shows the range with some context instead of the whole video (needed for long recordings).
Item {
    id: root
    property real duration: 0
    property real position: 0
    property real rangeStart: 0
    property real rangeEnd: 0
    property bool scrubEnabled: true
    property bool zoomed: false
    property var formatTime: (t) => t.toFixed(1)
    readonly property real zoomPadding: Math.max(2, (rangeEnd - rangeStart) * 0.25)
    readonly property real liveViewStart: zoomed ? Math.max(0, rangeStart - zoomPadding) : 0
    readonly property real liveViewEnd: zoomed ? Math.min(duration, rangeEnd + zoomPadding) : duration
    // while a range handle is dragged the view stays put, otherwise a zoomed view would chase the handle
    property bool viewFrozen: false
    property real frozenStart: 0
    property real frozenEnd: 0
    readonly property real viewStart: viewFrozen ? frozenStart : liveViewStart
    readonly property real viewEnd: viewFrozen ? frozenEnd : liveViewEnd
    signal seekRequested(real t)
    signal rangeStartRequested(real t)
    signal rangeEndRequested(real t)

    implicitHeight: 58
    readonly property real trackX: 10
    readonly property real trackW: Math.max(1, width - 2 * trackX)

    function xFor(t) {
        const span = viewEnd - viewStart
        return trackX + (span > 0 ? Math.min(Math.max((t - viewStart) / span, 0), 1) : 0) * trackW
    }
    function tFor(x) {
        return viewStart + Math.min(Math.max((x - trackX) / trackW, 0), 1) * (viewEnd - viewStart)
    }

    Rectangle {
        id: track
        x: root.trackX
        width: root.trackW
        y: 16
        height: 8
        radius: 4
        color: Theme.surfaceRaised
        border.color: Theme.border
    }

    Rectangle {
        id: rangeBar
        x: root.xFor(root.rangeStart)
        width: Math.max(2, root.xFor(root.rangeEnd) - x)
        y: track.y
        height: track.height
        radius: 4
        color: Theme.withAlpha(Theme.accent, 0.35)
        border.color: Theme.accent
    }

    MouseArea {
        anchors.fill: parent
        enabled: root.scrubEnabled && root.duration > 0
        cursorShape: enabled ? Qt.PointingHandCursor : Qt.ArrowCursor
        onPressed: (mouse) => root.seekRequested(root.tFor(mouse.x))
        onPositionChanged: (mouse) => { if (pressed) root.seekRequested(root.tFor(mouse.x)) }
    }

    Repeater {
        model: [{ start: true }, { start: false }]
        delegate: Rectangle {
            required property var modelData
            readonly property bool isStart: modelData.start
            x: root.xFor(isStart ? root.rangeStart : root.rangeEnd) - width / 2
            y: 8
            width: 10
            height: 24
            radius: 3
            color: handleArea.containsMouse || handleArea.pressed ? Qt.lighter(Theme.accent, 1.2) : Theme.accent
            border.color: Theme.bg
            border.width: 2
            ToolTip.visible: handleArea.containsMouse || handleArea.pressed
            ToolTip.text: isStart ? "Range start" : "Range end"
            MouseArea {
                id: handleArea
                anchors.fill: parent
                anchors.margins: -6
                hoverEnabled: true
                enabled: root.duration > 0
                preventStealing: true
                cursorShape: Qt.SizeHorCursor
                onPressed: {
                    root.frozenStart = root.liveViewStart
                    root.frozenEnd = root.liveViewEnd
                    root.viewFrozen = true
                }
                onReleased: root.viewFrozen = false
                onCanceled: root.viewFrozen = false
                onPositionChanged: (mouse) => {
                    if (!pressed)
                        return
                    const t = root.tFor(mapToItem(root, mouse.x, mouse.y).x)
                    if (parent.isStart) root.rangeStartRequested(t); else root.rangeEndRequested(t)
                }
            }
        }
    }

    Rectangle {
        id: playhead
        x: root.xFor(root.position) - 1
        y: 2
        width: 2
        height: 36
        radius: 1
        color: Theme.text
        Rectangle {
            anchors.horizontalCenter: parent.horizontalCenter
            y: -1
            width: 10
            height: 10
            radius: 5
            color: Theme.text
        }
    }

    Label {
        x: root.trackX
        y: 42
        text: root.formatTime(root.viewStart)
        color: Theme.textFaint
        font.pixelSize: 11
    }
    Label {
        x: root.trackX + root.trackW - width
        y: 42
        text: root.formatTime(root.viewEnd)
        color: Theme.textFaint
        font.pixelSize: 11
    }
}
