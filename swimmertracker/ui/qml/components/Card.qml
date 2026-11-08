import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../theme"

// Rounded surface with an optional uppercase title and a right-aligned header slot.
Rectangle {
    id: root
    property string title: ""
    property alias headerItems: headerSlot.data
    default property alias content: body.data
    property int padding: 14

    color: Theme.surface
    radius: Theme.radius
    border.color: Theme.border
    border.width: 1
    implicitHeight: column.implicitHeight + 2 * padding
    implicitWidth: column.implicitWidth + 2 * padding

    ColumnLayout {
        id: column
        anchors.fill: parent
        anchors.margins: root.padding
        spacing: 10

        RowLayout {
            visible: root.title !== "" || headerSlot.children.length > 0
            Layout.fillWidth: true
            spacing: 8
            Label {
                text: root.title
                color: Theme.textDim
                font.pixelSize: 12
                font.weight: Font.DemiBold
                font.capitalization: Font.AllUppercase
                font.letterSpacing: 0.8
            }
            Item { Layout.fillWidth: true }
            RowLayout { id: headerSlot; spacing: 6 }
        }

        ColumnLayout {
            id: body
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 8
        }
    }
}
