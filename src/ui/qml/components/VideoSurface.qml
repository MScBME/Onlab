import QtQuick
import QtMultimedia
import "../theme"

// Shows frames pushed from Python into its QVideoSink and maps image <-> item coordinates
// for the overlay children (declared inside a VideoSurface).
Item {
    id: root
    property int frameWidth: 1
    property int frameHeight: 1
    property alias fillMode: output.fillMode
    readonly property rect contentRect: output.contentRect
    readonly property bool hasContent: output.sourceRect.width > 0 && contentRect.width > 1
    readonly property var videoSink: output.videoSink
    default property alias overlay: overlayLayer.data
    signal sinkReady(var sink)

    function toItem(px, py) {
        return Qt.point(contentRect.x + px * contentRect.width / Math.max(1, frameWidth),
                        contentRect.y + py * contentRect.height / Math.max(1, frameHeight))
    }

    function toImage(ix, iy) {
        return Qt.point((ix - contentRect.x) * frameWidth / Math.max(1, contentRect.width),
                        (iy - contentRect.y) * frameHeight / Math.max(1, contentRect.height))
    }

    Rectangle {
        anchors.fill: parent
        color: Theme.videoBg
        radius: Theme.radiusSmall
    }

    VideoOutput {
        id: output
        anchors.fill: parent
        fillMode: VideoOutput.PreserveAspectFit
    }

    Item {
        id: overlayLayer
        anchors.fill: parent
    }

    Component.onCompleted: sinkReady(output.videoSink)
}
