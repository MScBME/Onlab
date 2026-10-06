import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs
import "../theme"

// Draw and edit lane quads on any frame of the video (backed by the `laneEditor` controller).
RowLayout {
    id: root
    spacing: Theme.gap
    property bool addMode: false
    property var pendingPoints: []

    function cancelAdd() {
        addMode = false
        pendingPoints = []
    }

    // Called by the hosting dialog when it opens: both dialogs embed an editor view, but there is
    // only one editor controller, so the visible view claims the video sinks.
    function attachSinks() {
        cancelAdd()
        laneEditor.setFrameSink(frameView.videoSink)
        laneEditor.setWarpSink(warpView.videoSink)
    }

    Shortcut {
        sequence: "Escape"
        enabled: root.visible && root.addMode
        onActivated: root.cancelAdd()
    }

    // ---- frame + overlays ------------------------------------------------------------------
    ColumnLayout {
        Layout.fillWidth: true
        Layout.fillHeight: true
        spacing: 10

        VideoSurface {
            id: frameView
            Layout.fillWidth: true
            Layout.fillHeight: true
            frameWidth: laneEditor.frameWidth
            frameHeight: laneEditor.frameHeight
            
            Repeater {
                model: laneEditor.lanes
                delegate: LanePolygon {
                    required property int index
                    required laneId
                    required points
                    required lengthM
                    required startLabel
                    required endLabel
                    surface: frameView
                    highlighted: index === laneEditor.selectedIndex
                }
            }

            // corner handles of the selected lane (fixed set of 4 items, so a drag survives model updates)
            Repeater {
                model: 4
                delegate: Rectangle {
                    id: handle
                    required property int index
                    readonly property var pt: laneEditor.selectedPoints.length === 4 ? laneEditor.selectedPoints[index] : null
                    readonly property point pos: pt && frameView.hasContent ? frameView.toItem(pt[0], pt[1]) : Qt.point(-100, -100)
                    visible: pt !== null && !root.addMode && frameView.hasContent
                    x: pos.x - width / 2
                    y: pos.y - height / 2
                    width: 16
                    height: 16
                    radius: 8
                    color: dragArea.pressed || dragArea.containsMouse ? Theme.text : Theme.bg
                    border.color: Theme.laneColor(laneEditor.selectedLaneId)
                    border.width: 3
                    MouseArea {
                        id: dragArea
                        anchors.fill: parent
                        anchors.margins: -8
                        hoverEnabled: true
                        preventStealing: true
                        cursorShape: Qt.SizeAllCursor
                        onPositionChanged: (mouse) => {
                            if (!pressed)
                                return
                            const p = mapToItem(frameView, mouse.x, mouse.y)
                            const img = frameView.toImage(p.x, p.y)
                            laneEditor.moveCorner(laneEditor.selectedIndex, handle.index, img.x, img.y)
                        }
                    }
                }
            }

            // add-lane mode: collect 4 clicks
            MouseArea {
                anchors.fill: parent
                enabled: root.addMode
                visible: root.addMode
                cursorShape: Qt.CrossCursor
                onClicked: (mouse) => {
                    const img = frameView.toImage(mouse.x, mouse.y)
                    if (img.x < 0 || img.y < 0 || img.x > laneEditor.frameWidth || img.y > laneEditor.frameHeight)
                        return
                    const pts = root.pendingPoints.concat([[img.x, img.y]])
                    if (pts.length === 4) {
                        laneEditor.addLane(pts)
                        root.cancelAdd()
                    } else {
                        root.pendingPoints = pts
                    }
                }
            }

            Repeater {
                model: root.pendingPoints
                delegate: Rectangle {
                    required property var modelData
                    required property int index
                    readonly property point pos: frameView.toItem(modelData[0], modelData[1])
                    x: pos.x - 7
                    y: pos.y - 7
                    width: 14
                    height: 14
                    radius: 7
                    color: Theme.accent
                    border.color: Theme.bg
                    border.width: 2
                    Label {
                        anchors.left: parent.right
                        anchors.leftMargin: 4
                        anchors.verticalCenter: parent.verticalCenter
                        text: index + 1
                        color: Theme.accent
                        font.weight: Font.DemiBold
                    }
                }
            }

            Rectangle {
                visible: root.addMode
                anchors.top: parent.top
                anchors.horizontalCenter: parent.horizontalCenter
                anchors.topMargin: 12
                width: addHint.implicitWidth + 24
                height: 32
                radius: 16
                color: Theme.withAlpha(Theme.bg, 0.9)
                border.color: Theme.accent
                Label {
                    id: addHint
                    anchors.centerIn: parent
                    text: `Click the 4 corners of the lane (${root.pendingPoints.length}/4) · Esc to cancel`
                    color: Theme.text
                }
            }
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 10
            Label {
                text: "Frame"
                color: Theme.textDim
            }
            Slider {
                id: frameSlider
                Layout.fillWidth: true
                from: 0
                to: Math.max(0.1, laneEditor.duration)
                onMoved: laneEditor.seek(value)
                Binding on value {
                    value: laneEditor.frameTime
                    when: !frameSlider.pressed
                    restoreMode: Binding.RestoreNone
                }
            }
            Label {
                text: workspace.formatTimeMs(laneEditor.frameTime)
                color: Theme.text
                font.family: Theme.mono
            }
        }
        Label {
            text: root.addMode ? "Corners can be clicked in any order; they are ordered automatically."
                               : "Pick a frame where the lane is clearly visible. Drag the handles to adjust the selected lane."
            color: Theme.textFaint
            font.pixelSize: 12
        }
    }

    // ---- side panel --------------------------------------------------------------------
    ColumnLayout {
        Layout.preferredWidth: 380
        Layout.maximumWidth: 380
        Layout.fillHeight: true
        spacing: 10

        Button {
            Layout.fillWidth: true
            text: root.addMode ? "Cancel adding" : "+  Add lane"
            highlighted: !root.addMode
            onClicked: root.addMode ? root.cancelAdd() : (root.addMode = true)
        }

        RowLayout {
            Layout.fillWidth: true
            visible: laneEditor.copySources.length > 0
            ComboBox {
                id: copySource
                Layout.fillWidth: true
                // index 0 is a placeholder, so a model reset never pre-selects a real video
                model: ["Copy lanes from…"].concat(laneEditor.copySources)
                displayText: currentIndex > 0 ? `Copy from: ${currentText}` : currentText
            }
            Button {
                text: "Copy"
                enabled: copySource.currentIndex > 0
                onClicked: copyConfirm.open()
            }
        }

        MessageDialog {
            id: copyConfirm
            title: "Copy lanes"
            text: `Replace all lanes with the lanes of '${copySource.currentText}'?`
            buttons: MessageDialog.Yes | MessageDialog.Cancel
            onButtonClicked: (button) => {
                if (button === MessageDialog.Yes) {
                    laneEditor.copyLanesFrom(copySource.currentText)
                    copySource.currentIndex = 0
                }
            }
        }

        Label {
            text: `Lanes (${laneEditor.lanes.count})`
            color: Theme.textDim
            font.pixelSize: 12
            font.weight: Font.DemiBold
            font.capitalization: Font.AllUppercase
        }

        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 10

            ListView {
                id: laneList
                Layout.fillWidth: true
                Layout.fillHeight: true
                clip: true
                spacing: 6
                model: laneEditor.lanes
                ScrollBar.vertical: ScrollBar {}

                delegate: Rectangle {
                    id: laneRow
                    required property int index
                    required property int laneId
                    required property real lengthM
                    required property bool isNew
                    required property bool deletable
                    required property string refText
                    required property bool valid
                    readonly property bool selected: index === laneEditor.selectedIndex
                    width: ListView.view.width
                    height: rowLayout.implicitHeight + 16
                    radius: Theme.radiusSmall
                    color: selected ? Theme.surfaceHover : Theme.surfaceRaised
                    border.color: selected ? Theme.laneColor(laneId) : Theme.border

                    MouseArea {
                        anchors.fill: parent
                        onClicked: laneEditor.selectLane(laneRow.index)
                    }

                    ColumnLayout {
                        id: rowLayout
                        anchors.fill: parent
                        anchors.margins: 8
                        spacing: 6
                        RowLayout {
                            spacing: 8
                            Rectangle {
                                width: 10
                                height: 10
                                radius: 5
                                color: Theme.laneColor(laneRow.laneId)
                            }
                            Label {
                                text: "Lane"
                                color: Theme.text
                            }
                            SpinBox {
                                visible: laneRow.isNew
                                from: 1
                                to: laneEditor.maxLaneId
                                value: laneRow.laneId
                                editable: true
                                Layout.preferredWidth: 110
                                onValueModified: laneEditor.setLaneId(laneRow.index, value)
                            }
                            Label {
                                visible: !laneRow.isNew
                                text: laneRow.laneId
                                color: Theme.text
                                font.weight: Font.DemiBold
                            }
                            Label {
                                visible: !laneRow.valid
                                text: "invalid"
                                color: Theme.danger
                                font.pixelSize: 11
                            }
                            Item { Layout.fillWidth: true }
                            ToolButton {
                                text: "🗑"
                                enabled: laneRow.deletable
                                onClicked: laneEditor.removeLane(laneRow.index)
                                ToolTip.visible: hovered
                                ToolTip.text: laneRow.deletable ? "Remove lane" : laneRow.refText
                                hoverEnabled: true
                            }
                        }
                        RowLayout {
                            spacing: 8
                            Label {
                                text: "Length"
                                color: Theme.textDim
                            }
                            TextField {
                                Layout.preferredWidth: 80
                                text: Number(laneRow.lengthM).toString()
                                horizontalAlignment: Text.AlignRight
                                validator: DoubleValidator {
                                    bottom: 0.1
                                    top: 200
                                    decimals: 2
                                    notation: DoubleValidator.StandardNotation
                                    locale: "C"
                                }
                                onEditingFinished: laneEditor.setLaneLength(laneRow.index, text)
                            }
                            Label {
                                text: "m"
                                color: Theme.textDim
                            }
                            Item { Layout.fillWidth: true }
                            Label {
                                visible: !laneRow.deletable
                                text: "in use"
                                color: Theme.textFaint
                                font.pixelSize: 11
                                ToolTip.visible: inUseHover.hovered
                                ToolTip.text: laneRow.refText
                                HoverHandler { id: inUseHover }
                            }
                        }
                    }
                }

                Label {
                    anchors.centerIn: parent
                    visible: laneList.count === 0
                    width: parent.width - 20
                    wrapMode: Text.Wrap
                    horizontalAlignment: Text.AlignHCenter
                    text: "No lanes yet.\nAdd at least one lane to continue."
                    color: Theme.textFaint
                }
            }

            ColumnLayout {
                Layout.fillWidth: false
                Layout.preferredWidth: 104
                Layout.maximumWidth: 104
                Layout.fillHeight: true
                spacing: 4
                Label {
                    text: "Bird's-eye"
                    color: Theme.textFaint
                    font.pixelSize: 10
                    Layout.alignment: Qt.AlignHCenter
                }
                VideoSurface {
                    id: warpView
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    frameWidth: 256
                    frameHeight: 1024
                }
            }
        }

        Label {
            Layout.fillWidth: true
            visible: laneEditor.notice !== ""
            text: laneEditor.notice
            color: Theme.warning
            wrapMode: Text.Wrap
            font.pixelSize: 12
        }

        Repeater {
            model: laneEditor.errors
            delegate: Label {
                required property string modelData
                Layout.fillWidth: true
                text: "• " + modelData
                color: Theme.danger
                wrapMode: Text.Wrap
                font.pixelSize: 12
            }
        }
    }
}
