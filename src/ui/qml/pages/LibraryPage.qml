import QtQuick
import QtQuick.Controls
import QtQuick.Effects
import QtQuick.Layouts
import "../theme"
import "../components"

Page {
    id: page
    signal openVideo(string videoId)
    signal addVideo()
    signal editLanes(string videoId)

    background: Rectangle { color: Theme.bg }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 28
        spacing: 22

        RowLayout {
            Layout.fillWidth: true
            spacing: 12
            ColumnLayout {
                spacing: 2
                Label {
                    text: "Video library"
                    color: Theme.text
                    font.pixelSize: 28
                    font.weight: Font.DemiBold
                }
                Label {
                    text: `${library.videos.count} ${library.videos.count === 1 ? "video" : "videos"} · files live in data/raw`
                    color: Theme.textDim
                }
            }
            Item { Layout.fillWidth: true }
            Button {
                text: "Refresh"
                flat: true
                onClicked: library.refresh()
            }
            Button {
                text: "+  Add video"
                highlighted: true
                onClicked: page.addVideo()
            }
        }

        GridView {
            id: grid
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true
            readonly property int columns: Math.max(1, Math.floor(width / 330))
            cellWidth: Math.floor(width / columns)
            cellHeight: Math.round((cellWidth - 16) * 9 / 16) + 118
            model: library.videos
            ScrollBar.vertical: ScrollBar {}

            delegate: Item {
                id: cell
                required property string videoId
                required property string fileName
                required property string durationText
                required property string detailText
                required property int laneCount
                required property int clipCount
                required property bool missing
                required property string thumbnail
                width: grid.cellWidth
                height: grid.cellHeight

                Rectangle {
                    id: card
                    anchors.fill: parent
                    anchors.margins: 8
                    radius: Theme.radius
                    color: hover.hovered && !cell.missing ? Theme.surfaceHover : Theme.surface
                    border.color: hover.hovered && !cell.missing ? Theme.borderStrong : Theme.border
                    Behavior on color { ColorAnimation { duration: 120 } }

                    HoverHandler { id: hover }
                    // below the content, so clicks on the card's own buttons don't also open the video
                    MouseArea {
                        anchors.fill: parent
                        enabled: !cell.missing
                        cursorShape: enabled ? Qt.PointingHandCursor : Qt.ArrowCursor
                        onClicked: page.openVideo(cell.videoId)
                    }

                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: 8
                        spacing: 8

                        Item {
                            Layout.fillWidth: true
                            Layout.preferredHeight: Math.round(width * 9 / 16)

                            Image {
                                id: thumb
                                anchors.fill: parent
                                source: cell.thumbnail
                                asynchronous: true
                                fillMode: Image.PreserveAspectCrop
                                visible: false
                            }
                            Rectangle {
                                id: thumbMask
                                anchors.fill: parent
                                radius: Theme.radiusSmall
                                visible: false
                                layer.enabled: true
                            }
                            Rectangle {
                                anchors.fill: parent
                                radius: Theme.radiusSmall
                                color: Theme.videoBg
                            }
                            MultiEffect {
                                anchors.fill: thumb
                                source: thumb
                                maskEnabled: true
                                maskSource: thumbMask
                                opacity: thumb.status === Image.Ready ? (cell.missing ? 0.35 : 1) : 0
                                Behavior on opacity { NumberAnimation { duration: 200 } }
                            }
                            BusyIndicator {
                                anchors.centerIn: parent
                                running: thumb.status === Image.Loading
                                visible: running
                            }
                            Label {
                                anchors.centerIn: parent
                                visible: cell.missing
                                text: "File missing"
                                color: Theme.danger
                                font.weight: Font.DemiBold
                            }
                            Rectangle {
                                visible: cell.durationText !== ""
                                anchors.right: parent.right
                                anchors.bottom: parent.bottom
                                anchors.margins: 8
                                width: durationLabel.implicitWidth + 12
                                height: 22
                                radius: 4
                                color: Theme.withAlpha(Theme.bg, 0.8)
                                Label {
                                    id: durationLabel
                                    anchors.centerIn: parent
                                    text: cell.durationText
                                    color: Theme.text
                                    font.pixelSize: 12
                                    font.features: { "tnum": 1 }
                                }
                            }
                        }

                        RowLayout {
                            Layout.fillWidth: true
                            Layout.leftMargin: 4
                            Layout.rightMargin: 2
                            spacing: 6
                            ColumnLayout {
                                Layout.fillWidth: true
                                spacing: 1
                                Label {
                                    Layout.fillWidth: true
                                    text: cell.videoId
                                    color: Theme.text
                                    font.pixelSize: 15
                                    font.weight: Font.DemiBold
                                    elide: Text.ElideRight
                                }
                                Label {
                                    Layout.fillWidth: true
                                    text: cell.fileName
                                    color: Theme.textDim
                                    font.pixelSize: 12
                                    elide: Text.ElideMiddle
                                }
                            }
                            ToolButton {
                                text: "Edit lanes"
                                enabled: !cell.missing
                                font.pixelSize: 12
                                onClicked: page.editLanes(cell.videoId)
                            }
                        }

                        RowLayout {
                            Layout.leftMargin: 4
                            spacing: 6
                            Repeater {
                                model: [
                                    `${cell.laneCount} ${cell.laneCount === 1 ? "lane" : "lanes"}`,
                                    `${cell.clipCount} ${cell.clipCount === 1 ? "clip" : "clips"}`
                                ]
                                delegate: Rectangle {
                                    required property string modelData
                                    implicitWidth: chipText.implicitWidth + 14
                                    implicitHeight: 22
                                    radius: 11
                                    color: Theme.surfaceRaised
                                    border.color: Theme.border
                                    Label {
                                        id: chipText
                                        anchors.centerIn: parent
                                        text: modelData
                                        color: Theme.textDim
                                        font.pixelSize: 11
                                    }
                                }
                            }
                            Label {
                                Layout.leftMargin: 4
                                text: cell.missing ? "" : cell.detailText
                                color: Theme.textFaint
                                font.pixelSize: 11
                            }
                        }
                        Item { Layout.fillHeight: true }
                    }
                }
            }

            ColumnLayout {
                anchors.centerIn: parent
                visible: grid.count === 0
                spacing: 12
                Label {
                    Layout.alignment: Qt.AlignHCenter
                    text: "No videos yet"
                    color: Theme.text
                    font.pixelSize: 20
                }
                Label {
                    Layout.alignment: Qt.AlignHCenter
                    text: "Add a video and mark at least one lane to start analyzing."
                    color: Theme.textDim
                }
                Button {
                    Layout.alignment: Qt.AlignHCenter
                    text: "+  Add video"
                    highlighted: true
                    onClicked: page.addVideo()
                }
            }
        }
    }
}
