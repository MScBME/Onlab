pragma Singleton
import QtQuick

QtObject {
    readonly property color bg: "#0e1115"
    readonly property color surface: "#161a20"
    readonly property color surfaceRaised: "#1d2229"
    readonly property color surfaceHover: "#232932"
    readonly property color border: "#262c35"
    readonly property color borderStrong: "#353d48"
    readonly property color text: "#e7eaef"
    readonly property color textDim: "#9aa3ae"
    readonly property color textFaint: "#69727e"
    readonly property color accent: "#4cc2ff"
    readonly property color success: "#3ddc97"
    readonly property color warning: "#ffb547"
    readonly property color danger: "#ff6b6b"
    readonly property color videoBg: "#07080a"

    readonly property int radius: 10
    readonly property int radiusSmall: 6
    readonly property int gap: 14
    readonly property string mono: "Cascadia Mono, Consolas"

    readonly property var lanePalette: ["#4cc2ff", "#ff9f43", "#a78bfa", "#3ddc97", "#ff6b9a",
                                        "#ffd166", "#5eead4", "#f97316", "#93c5fd", "#c4b5fd"]

    function laneColor(laneId) {
        const n = lanePalette.length
        return lanePalette[((laneId - 1) % n + n) % n]
    }

    function withAlpha(c, a) {
        return Qt.rgba(c.r, c.g, c.b, a)
    }

    function stateLabel(state) {
        switch (state) {
        case "preview": return "Previewing"
        case "loading": return "Loading model"
        case "analyzing": return "Analyzing"
        case "paused": return "Paused"
        case "finished": return "Finished"
        case "error": return "Error"
        default: return "Ready"
        }
    }

    function stateColor(state) {
        switch (state) {
        case "analyzing": return success
        case "paused": return warning
        case "loading": return accent
        case "finished": return accent
        case "error": return danger
        default: return textDim
        }
    }

    function detectionLabel(status) {
        switch (status) {
        case "detected": return "Detected"
        case "predicted": return "Predicted (Kalman)"
        case "searching": return "Searching"
        default: return "–"
        }
    }

    function detectionColor(status) {
        switch (status) {
        case "detected": return success
        case "predicted": return warning
        case "searching": return danger
        default: return textDim
        }
    }
}
