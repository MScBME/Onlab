import QtQuick
import QtQuick.Controls
import QtQuick.Shapes
import "../theme"

// One lane quad drawn over a VideoSurface (points are in image pixels).
Item {
    id: root
    anchors.fill: parent
    property var surface
    property var points: []
    property int laneId: 0
    property real lengthM: 25
    property bool highlighted: false
    property bool showBadge: true
    property var startLabel: []   // optional [x, y]: where "0 m" is drawn
    property var endLabel: []     // optional [x, y]: where "<length> m" is drawn
    readonly property color laneColor: Theme.laneColor(laneId)
    readonly property bool drawable: surface && surface.hasContent && points && points.length === 4

    function mapped() {
        if (!drawable)
            return []
        const out = []
        for (let i = 0; i < 4; ++i)
            out.push(surface.toItem(points[i][0], points[i][1]))
        out.push(out[0])
        return out
    }

    function at(p) {
        return drawable && p && p.length === 2 ? surface.toItem(p[0], p[1]) : Qt.point(-1000, -1000)
    }

    Shape {
        anchors.fill: parent
        visible: root.drawable
        preferredRendererType: Shape.CurveRenderer
        ShapePath {
            strokeColor: root.laneColor
            strokeWidth: root.highlighted ? 2.5 : 1.2
            fillColor: Theme.withAlpha(root.laneColor, root.highlighted ? 0.16 : 0.05)
            joinStyle: ShapePath.RoundJoin
            PathPolyline { path: root.mapped() }
        }
    }

    Rectangle {
        id: badge
        visible: root.drawable && root.showBadge
        readonly property point anchorPt: root.drawable
            ? root.surface.toItem((root.points[0][0] + root.points[1][0]) / 2, (root.points[0][1] + root.points[1][1]) / 2)
            : Qt.point(0, 0)
        x: Math.max(2, anchorPt.x - width - 6)
        y: anchorPt.y - height / 2
        width: badgeText.implicitWidth + 12
        height: 20
        radius: 10
        color: root.highlighted ? root.laneColor : Theme.withAlpha(Theme.bg, 0.8)
        border.color: root.laneColor
        Label {
            id: badgeText
            anchors.centerIn: parent
            text: root.highlighted ? `Lane ${root.laneId}` : `${root.laneId}`
            color: root.highlighted ? Theme.bg : root.laneColor
            font.pixelSize: 11
            font.weight: Font.DemiBold
        }
    }

    Repeater {
        model: root.highlighted ? [[root.startLabel, "0 m"], [root.endLabel, `${root.lengthM} m`]] : []
        delegate: Rectangle {
            required property var modelData
            readonly property point p: root.at(modelData[0])
            visible: modelData[0] && modelData[0].length === 2
            x: p.x - width / 2
            y: p.y - height - 8
            width: endText.implicitWidth + 10
            height: 18
            radius: 4
            color: Theme.withAlpha(Theme.bg, 0.85)
            border.color: root.laneColor
            Label {
                id: endText
                anchors.centerIn: parent
                text: modelData[1]
                color: Theme.text
                font.pixelSize: 10
            }
        }
    }
}
