import QtQuick
import QtGraphs
import "../theme"

// Speed over time: live Kalman samples while analyzing, smoothed profile when the run ends.
GraphsView {
    id: chart
    property real xMax: 10
    property real yMax: 2.5

    function reset() {
        live.clear()
        smooth.clear()
        yMax = 2.5
    }

    function appendLive(points) {
        for (const p of points) {
            live.append(p[0], p[1])
            if (p[1] > yMax * 0.92)
                yMax = Math.ceil(p[1] * 1.25 * 2) / 2
        }
    }

    function setSmoothed(points) {
        smooth.clear()
        for (const p of points)
            smooth.append(p[0], p[1])
    }

    marginTop: 8
    marginBottom: 4
    marginLeft: 4
    marginRight: 12

    theme: GraphsTheme {
        colorScheme: GraphsTheme.ColorScheme.Dark
        backgroundVisible: false
        plotAreaBackgroundVisible: false
        labelTextColor: Theme.textDim
        grid.mainColor: Theme.border
        grid.subColor: Theme.withAlpha(Theme.border, 0.4)
        axisX.mainColor: Theme.borderStrong
        axisY.mainColor: Theme.borderStrong
    }

    axisX: ValueAxis {
        min: 0
        max: Math.max(1, chart.xMax)
        labelDecimals: 0
        subTickCount: 0
        titleText: "Time in range (s)"
        titleColor: Theme.textDim
        titleFont.pixelSize: 11
    }

    axisY: ValueAxis {
        min: 0
        max: chart.yMax
        labelDecimals: 1
        tickInterval: chart.yMax > 3 ? 1 : 0.5
        subTickCount: 0
        titleText: "Speed (m/s)"
        titleColor: Theme.textDim
        titleFont.pixelSize: 11
    }

    LineSeries {
        id: live
        color: Theme.withAlpha(Theme.accent, 0.85)
        width: 1.5
    }

    LineSeries {
        id: smooth
        color: Theme.success
        width: 2.5
    }
}
