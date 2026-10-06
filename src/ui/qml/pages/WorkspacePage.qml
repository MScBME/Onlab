import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../theme"
import "../components"

Page {
    id: page
    signal back()
    signal editLanes()
    signal createClip()
    property bool shortcutsEnabled: true

    readonly property string playerState: workspace.state
    readonly property bool running: playerState === "loading" || playerState === "analyzing" || playerState === "paused"
    readonly property var currentLane: {
        const lanes = workspace.lanes
        for (let i = 0; i < lanes.length; ++i)
            if (lanes[i].id === workspace.laneId)
                return lanes[i]
        return null
    }

    background: Rectangle { color: Theme.bg }

    // ---- header ------------------------------------------------------------------------
    header: Rectangle {
        height: 64
        color: Theme.surface
        Rectangle {
            anchors.bottom: parent.bottom
            width: parent.width
            height: 1
            color: Theme.border
        }
        RowLayout {
            anchors.fill: parent
            anchors.leftMargin: 12
            anchors.rightMargin: 16
            spacing: 12
            ToolButton {
                text: "‹  Library"
                onClicked: page.back()
            }
            Rectangle {
                width: 1
                Layout.preferredHeight: 28
                color: Theme.border
            }
            ColumnLayout {
                spacing: 0
                Label {
                    text: workspace.videoId
                    color: Theme.text
                    font.pixelSize: 17
                    font.weight: Font.DemiBold
                }
                Label {
                    text: workspace.frameWidth > 0
                          ? `${workspace.fileName} · ${workspace.frameWidth}×${workspace.frameHeight} · ${workspace.fps.toFixed(0)} fps · ${workspace.formatTime(workspace.duration)}`
                          : workspace.fileName
                    color: Theme.textDim
                    font.pixelSize: 12
                }
            }
            Item { Layout.fillWidth: true }
            Rectangle {
                // implicit size: the RowLayout re-flows when the label changes ("Ready" -> "Loading model")
                implicitHeight: 26
                implicitWidth: stateRow.implicitWidth + 20
                radius: 13
                color: Theme.withAlpha(Theme.stateColor(page.playerState), 0.14)
                border.color: Theme.withAlpha(Theme.stateColor(page.playerState), 0.5)
                RowLayout {
                    id: stateRow
                    anchors.centerIn: parent
                    spacing: 6
                    Rectangle {
                        width: 8
                        height: 8
                        radius: 4
                        color: Theme.stateColor(page.playerState)
                        SequentialAnimation on opacity {
                            running: page.playerState === "analyzing" || page.playerState === "loading"
                            loops: Animation.Infinite
                            NumberAnimation { to: 0.3; duration: 600 }
                            NumberAnimation { to: 1; duration: 600 }
                            onRunningChanged: if (!running) parent.opacity = 1
                        }
                    }
                    Label {
                        text: Theme.stateLabel(page.playerState)
                        color: Theme.stateColor(page.playerState)
                        font.pixelSize: 12
                        font.weight: Font.DemiBold
                    }
                }
            }
            Button {
                text: "Edit lanes"
                enabled: !page.running
                onClicked: page.editLanes()
            }
            Button {
                text: "Create clip…"
                enabled: workspace.laneId > 0 && workspace.rangeEnd > workspace.rangeStart
                onClicked: page.createClip()
            }
        }
    }

    // ---- body --------------------------------------------------------------------------
    RowLayout {
        anchors.fill: parent
        anchors.margins: 16
        spacing: 16

        ColumnLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 12

            RowLayout {
                Layout.fillWidth: true
                Layout.fillHeight: true
                spacing: 12

                VideoSurface {
                    id: original
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    frameWidth: workspace.frameWidth
                    frameHeight: workspace.frameHeight
                    onSinkReady: (sink) => workspace.setOriginalSink(sink)

                    Repeater {
                        model: workspace.lanes
                        delegate: LanePolygon {
                            required property var modelData
                            surface: original
                            laneId: modelData.id
                            points: modelData.points
                            lengthM: modelData.lengthM
                            highlighted: modelData.id === workspace.laneId
                        }
                    }

                    Rectangle {
                        readonly property bool shown: workspace.swimmerPoint.length === 2 && original.hasContent
                        readonly property point p: shown ? original.toItem(workspace.swimmerPoint[0], workspace.swimmerPoint[1]) : Qt.point(0, 0)
                        visible: shown
                        x: p.x - width / 2
                        y: p.y - height / 2
                        width: 18
                        height: 18
                        radius: 9
                        color: Theme.withAlpha(Theme.detectionColor(workspace.detectionStatus), 0.85)
                        border.color: "white"
                        border.width: 2
                    }
                }

                Card {
                    Layout.preferredWidth: 150
                    Layout.fillHeight: true
                    padding: 10
                    title: "Bird's-eye"

                    Label {
                        text: page.currentLane ? `Lane ${page.currentLane.id} · 0 m` : ""
                        color: Theme.textDim
                        font.pixelSize: 11
                        Layout.alignment: Qt.AlignHCenter
                    }
                    VideoSurface {
                        id: warped
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        frameWidth: 256
                        frameHeight: 1024
                        onSinkReady: (sink) => workspace.setWarpedSink(sink)

                        Rectangle {
                            readonly property bool shown: workspace.bbox.length === 4 && warped.hasContent
                            readonly property point a: shown ? warped.toItem(workspace.bbox[0], workspace.bbox[1]) : Qt.point(0, 0)
                            readonly property point b: shown ? warped.toItem(workspace.bbox[2], workspace.bbox[3]) : Qt.point(0, 0)
                            visible: shown
                            x: a.x
                            y: a.y
                            width: b.x - a.x
                            height: b.y - a.y
                            color: "transparent"
                            radius: 2
                            border.color: Theme.success
                            border.width: 2
                        }
                    }
                    Label {
                        text: page.currentLane ? `${page.currentLane.lengthM} m` : ""
                        color: Theme.textDim
                        font.pixelSize: 11
                        Layout.alignment: Qt.AlignHCenter
                    }
                }
            }

            // ---- timeline ----------------------------------------------------------------
            Card {
                Layout.fillWidth: true
                padding: 12

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 4
                    ToolButton {
                        text: "−1 s"
                        enabled: !page.running
                        onClicked: workspace.stepSeconds(-1)
                    }
                    ToolButton {
                        text: "◂"
                        enabled: !page.running
                        onClicked: workspace.stepFrames(-1)
                        ToolTip.visible: hovered
                        ToolTip.text: "Previous frame"
                    }
                    ToolButton {
                        text: page.playerState === "preview" ? "❚❚" : "▶"
                        enabled: !page.running
                        onClicked: workspace.togglePreview()
                        ToolTip.visible: hovered
                        ToolTip.text: "Play / pause preview (Space) — no detection"
                    }
                    ToolButton {
                        text: "▸"
                        enabled: !page.running
                        onClicked: workspace.stepFrames(1)
                        ToolTip.visible: hovered
                        ToolTip.text: "Next frame"
                    }
                    ToolButton {
                        text: "+1 s"
                        enabled: !page.running
                        onClicked: workspace.stepSeconds(1)
                    }
                    Label {
                        Layout.leftMargin: 10
                        text: workspace.formatTimeMs(workspace.position)
                        color: Theme.text
                        font.family: Theme.mono
                        font.pixelSize: 15
                    }
                    Label {
                        text: "/ " + workspace.formatTime(workspace.duration)
                        color: Theme.textFaint
                        font.family: Theme.mono
                    }
                    Item { Layout.fillWidth: true }
                    Label {
                        visible: page.running
                        text: "Scrubbing is disabled while an analysis is running"
                        color: Theme.textFaint
                        font.pixelSize: 12
                    }
                    Button {
                        id: zoomButton
                        objectName: "zoomButton"
                        checkable: true
                        flat: true
                        text: checked ? "Show full video" : "Zoom to range"
                    }
                }

                Timeline {
                    Layout.fillWidth: true
                    zoomed: zoomButton.checked
                    formatTime: (t) => workspace.formatTime(t)
                    duration: workspace.duration
                    position: workspace.position
                    rangeStart: workspace.rangeStart
                    rangeEnd: workspace.rangeEnd
                    scrubEnabled: !page.running
                    onSeekRequested: (t) => workspace.seek(t)
                    onRangeStartRequested: (t) => workspace.setRangeStart(t)
                    onRangeEndRequested: (t) => workspace.setRangeEnd(t)
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 8
                    Label {
                        text: "Start"
                        color: Theme.textDim
                    }
                    TextField {
                        id: startField
                        Layout.preferredWidth: 130
                        font.family: Theme.mono
                        Binding on text {
                            value: workspace.formatTimeMs(workspace.rangeStart)
                            when: !startField.activeFocus
                            restoreMode: Binding.RestoreNone
                        }
                        onEditingFinished: {
                            if (!workspace.setRangeStartText(text))
                                text = workspace.formatTimeMs(workspace.rangeStart)
                            focus = false
                        }
                    }
                    Button {
                        text: "Set start  [I]"
                        onClicked: workspace.setRangeStartToPosition()
                    }
                    Item { Layout.preferredWidth: 12 }
                    Label {
                        text: "End"
                        color: Theme.textDim
                    }
                    TextField {
                        id: endField
                        Layout.preferredWidth: 130
                        font.family: Theme.mono
                        Binding on text {
                            value: workspace.formatTimeMs(workspace.rangeEnd)
                            when: !endField.activeFocus
                            restoreMode: Binding.RestoreNone
                        }
                        onEditingFinished: {
                            if (!workspace.setRangeEndText(text))
                                text = workspace.formatTimeMs(workspace.rangeEnd)
                            focus = false
                        }
                    }
                    Button {
                        text: "Set end  [O]"
                        onClicked: workspace.setRangeEndToPosition()
                    }
                    Item { Layout.fillWidth: true }
                    Label {
                        text: `Range ${(workspace.rangeEnd - workspace.rangeStart).toFixed(1)} s`
                        color: Theme.textDim
                    }
                }
            }

            // ---- chart -------------------------------------------------------------------
            Card {
                Layout.fillWidth: true
                Layout.preferredHeight: 250
                title: "Speed"
                headerItems: [
                    Rectangle { width: 14; height: 3; radius: 1; color: Theme.accent },
                    Label { text: "Live (Kalman)"; color: Theme.textDim; font.pixelSize: 11 },
                    Item { width: 10; height: 1 },
                    Rectangle { width: 14; height: 3; radius: 1; color: Theme.success },
                    Label { text: "Smoothed (after the run)"; color: Theme.textDim; font.pixelSize: 11 }
                ]

                SpeedChart {
                    id: chart
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    xMax: workspace.chartDuration
                }
            }
        }

        // ---- sidebar -----------------------------------------------------------------------
        ScrollView {
            id: sidebar
            Layout.preferredWidth: 350
            Layout.fillHeight: true
            contentWidth: availableWidth
            clip: true

            ColumnLayout {
                width: sidebar.availableWidth
                spacing: 12

                Card {
                    Layout.fillWidth: true
                    title: "Analysis"

                    GridLayout {
                        Layout.fillWidth: true
                        columns: 2
                        columnSpacing: 10
                        rowSpacing: 8

                        Label { text: "Lane"; color: Theme.textDim }
                        ComboBox {
                            id: laneBox
                            Layout.fillWidth: true
                            model: workspace.laneIds
                            displayText: currentIndex >= 0 ? `Lane ${currentText}` : "No lanes"
                            onActivated: (i) => workspace.setLane(workspace.laneIds[i])
                            Binding {
                                target: laneBox
                                property: "currentIndex"
                                value: workspace.laneIds.indexOf(workspace.laneId)
                            }
                        }

                        Label { text: "Model"; color: Theme.textDim }
                        ComboBox {
                            id: modelBox
                            Layout.fillWidth: true
                            model: workspace.models
                            displayText: currentIndex >= 0 ? currentText.replace(/\.pt$/, "") : "No models in models/"
                            onActivated: (i) => workspace.setModel(workspace.models[i])
                            Binding {
                                target: modelBox
                                property: "currentIndex"
                                value: workspace.models.indexOf(workspace.modelName)
                            }
                        }

                        Label { text: "Confidence"; color: Theme.textDim }
                        RowLayout {
                            Layout.fillWidth: true
                            Slider {
                                id: confSlider
                                Layout.fillWidth: true
                                from: 0.05
                                to: 0.95
                                stepSize: 0.05
                                onMoved: workspace.setConfidence(value)
                                Binding {
                                    target: confSlider
                                    property: "value"
                                    value: workspace.confidence
                                }
                            }
                            Label {
                                text: workspace.confidence.toFixed(2)
                                color: Theme.text
                                font.family: Theme.mono
                            }
                        }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        Layout.topMargin: 4
                        spacing: 8
                        Button {
                            Layout.fillWidth: true
                            highlighted: true
                            text: page.playerState === "analyzing" ? "❚❚  Pause"
                                : page.playerState === "paused" ? "▶  Resume"
                                : page.playerState === "loading" ? "Loading…"
                                : "▶  Analyze"
                            enabled: page.playerState === "analyzing" || page.playerState === "paused" || workspace.canAnalyze
                            onClicked: workspace.primaryAction()
                        }
                        Button {
                            text: "⟲  Restart"
                            enabled: (page.running || page.playerState === "finished") && page.playerState !== "loading" && workspace.canAnalyze
                            highlighted: workspace.settingsDirty
                            onClicked: workspace.restart()
                        }
                        Button {
                            text: "■  Stop"
                            enabled: page.running
                            onClicked: workspace.stop()
                        }
                    }

                    Rectangle {
                        Layout.fillWidth: true
                        visible: workspace.settingsDirty
                        implicitHeight: dirtyLabel.implicitHeight + 16
                        radius: Theme.radiusSmall
                        color: Theme.withAlpha(Theme.warning, 0.12)
                        border.color: Theme.withAlpha(Theme.warning, 0.5)
                        Label {
                            id: dirtyLabel
                            anchors.fill: parent
                            anchors.margins: 8
                            text: "Settings changed — restart to apply."
                            color: Theme.warning
                            wrapMode: Text.Wrap
                        }
                    }

                    Label {
                        Layout.fillWidth: true
                        visible: page.playerState === "loading" && workspace.statusMessage !== ""
                        text: workspace.statusMessage
                        color: Theme.textDim
                        wrapMode: Text.Wrap
                        font.pixelSize: 12
                    }
                }

                Card {
                    Layout.fillWidth: true
                    title: "Live telemetry"

                    RowLayout {
                        spacing: 6
                        Label {
                            text: workspace.speed.toFixed(2)
                            color: Theme.text
                            font.pixelSize: 44
                            font.weight: Font.DemiBold
                            font.features: { "tnum": 1 }
                        }
                        Label {
                            text: "m/s"
                            color: Theme.textDim
                            font.pixelSize: 18
                            Layout.alignment: Qt.AlignBaseline
                        }
                    }

                    GridLayout {
                        Layout.fillWidth: true
                        columns: 3
                        columnSpacing: 16
                        rowSpacing: 10
                        StatTile {
                            label: "Position"
                            value: workspace.positionM >= 0 ? workspace.positionM.toFixed(1) : "–"
                            unit: workspace.positionM >= 0 ? "m" : ""
                        }
                        StatTile {
                            label: "Confidence"
                            value: workspace.detectionStatus === "detected" ? workspace.detectionConfidence.toFixed(2) : "–"
                        }
                        StatTile {
                            label: "Processing"
                            value: workspace.processingFps > 0 ? workspace.processingFps.toFixed(0) : "–"
                            unit: workspace.processingFps > 0 ? "fps" : ""
                        }
                        StatTile {
                            Layout.columnSpan: 2
                            label: "Status"
                            value: Theme.detectionLabel(workspace.detectionStatus)
                            valueColor: Theme.detectionColor(workspace.detectionStatus)
                            valueSize: 15
                        }
                        StatTile {
                            label: "Elapsed"
                            value: workspace.elapsed.toFixed(1)
                            unit: "s"
                        }
                    }
                }

                Card {
                    Layout.fillWidth: true
                    title: "Summary"
                    visible: workspace.summary.framesProcessed !== undefined

                    Label {
                        Layout.fillWidth: true
                        visible: workspace.summary.completed === false
                        text: `Stopped early at ${workspace.summary.endText || ""}`
                        color: Theme.warning
                        font.pixelSize: 12
                    }
                    GridLayout {
                        Layout.fillWidth: true
                        columns: 2
                        columnSpacing: 16
                        rowSpacing: 10
                        StatTile {
                            label: "Average speed"
                            value: workspace.summary.avgSpeed >= 0 ? workspace.summary.avgSpeed.toFixed(2) : "–"
                            unit: workspace.summary.avgSpeed >= 0 ? "m/s" : ""
                            valueSize: 22
                        }
                        StatTile {
                            label: "Max speed"
                            value: workspace.summary.maxSpeed >= 0 ? workspace.summary.maxSpeed.toFixed(2) : "–"
                            unit: workspace.summary.maxSpeed >= 0 ? "m/s" : ""
                            valueSize: 22
                        }
                        StatTile {
                            label: "Detection rate"
                            value: workspace.summary.detectionRate !== undefined ? (workspace.summary.detectionRate * 100).toFixed(0) : "–"
                            unit: "%"
                        }
                        StatTile {
                            label: "Analyzed"
                            value: workspace.summary.duration !== undefined ? workspace.summary.duration.toFixed(1) : "–"
                            unit: "s"
                        }
                    }
                    Label {
                        Layout.fillWidth: true
                        text: `${workspace.summary.framesDetected} of ${workspace.summary.framesProcessed} frames with a detection`
                        color: Theme.textFaint
                        font.pixelSize: 12
                    }
                    Label {
                        Layout.fillWidth: true
                        visible: workspace.summary.avgSpeed !== undefined && workspace.summary.avgSpeed < 0
                        text: "Not enough detections for speed statistics."
                        color: Theme.warning
                        wrapMode: Text.Wrap
                        font.pixelSize: 12
                    }
                }

                Card {
                    Layout.fillWidth: true
                    title: "Saved clips"

                    Label {
                        visible: workspace.clips.length === 0
                        text: "No clips for this video yet.\nUse “Create clip…” to save the current range."
                        color: Theme.textFaint
                        font.pixelSize: 12
                    }

                    Repeater {
                        model: workspace.clips
                        delegate: ItemDelegate {
                            required property var modelData
                            Layout.fillWidth: true
                            padding: 8
                            onClicked: workspace.applyClip(modelData.id)
                            ToolTip.visible: hovered
                            ToolTip.text: "Load this clip's lane and range"
                            contentItem: ColumnLayout {
                                spacing: 2
                                Label {
                                    Layout.fillWidth: true
                                    text: modelData.id
                                    color: Theme.text
                                    font.weight: Font.DemiBold
                                    elide: Text.ElideRight
                                }
                                Label {
                                    text: `Lane ${modelData.lane} · ${modelData.startText} – ${modelData.endText}`
                                    color: Theme.textDim
                                    font.pixelSize: 12
                                    font.family: Theme.mono
                                }
                                Label {
                                    Layout.fillWidth: true
                                    visible: modelData.description !== ""
                                    text: modelData.description
                                    color: Theme.textFaint
                                    font.pixelSize: 12
                                    elide: Text.ElideRight
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // ---- wiring -----------------------------------------------------------------------
    Timer {
        interval: 100
        repeat: true
        running: page.visible
        onTriggered: {
            const points = workspace.takeChartPoints()
            if (points.length)
                chart.appendLive(points)
        }
    }

    Connections {
        target: workspace
        function onChartReset() { chart.reset() }
        function onSummaryChanged() { chart.setSmoothed(workspace.smoothedSeries) }
    }

    Shortcut {
        sequence: "Space"
        enabled: page.shortcutsEnabled && !page.running
        onActivated: workspace.togglePreview()
    }
    Shortcut {
        sequence: "Left"
        enabled: page.shortcutsEnabled
        onActivated: workspace.stepSeconds(-1)
    }
    Shortcut {
        sequence: "Right"
        enabled: page.shortcutsEnabled
        onActivated: workspace.stepSeconds(1)
    }
    Shortcut {
        sequence: "Shift+Left"
        enabled: page.shortcutsEnabled
        onActivated: workspace.stepFrames(-1)
    }
    Shortcut {
        sequence: "Shift+Right"
        enabled: page.shortcutsEnabled
        onActivated: workspace.stepFrames(1)
    }
    Shortcut {
        sequence: "I"
        enabled: page.shortcutsEnabled
        onActivated: workspace.setRangeStartToPosition()
    }
    Shortcut {
        sequence: "O"
        enabled: page.shortcutsEnabled
        onActivated: workspace.setRangeEndToPosition()
    }
}
