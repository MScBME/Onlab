import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../theme"

ColumnLayout {
    id: root
    property string label: ""
    property string value: "–"
    property string unit: ""
    property color valueColor: Theme.text
    property int valueSize: 18
    spacing: 2

    Label {
        text: root.label
        color: Theme.textDim
        font.pixelSize: 11
    }
    RowLayout {
        spacing: 4
        Label {
            text: root.value
            color: root.valueColor
            font.pixelSize: root.valueSize
            font.weight: Font.DemiBold
            font.features: { "tnum": 1 }
        }
        Label {
            visible: root.unit !== ""
            text: root.unit
            color: Theme.textDim
            font.pixelSize: Math.max(11, root.valueSize * 0.55)
            Layout.alignment: Qt.AlignBaseline
        }
    }
}
