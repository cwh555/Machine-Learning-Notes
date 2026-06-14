import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import { pathToRoot, joinSegments } from "../util/path"
import { getPrivateBackgroundFiles } from "../util/privateBackgrounds"

const storageKey = "private-background"
const opacityStorageKey = "private-background-opacity"
const defaultOpacity = 15
const privateModeClass = "private-mode"
const changeEventName = "private-background-change"

const PrivateBackground: QuartzComponent = ({ fileData }: QuartzComponentProps) => {
  const backgrounds = getPrivateBackgroundFiles().map((background) => ({
    name: background.name,
    src: joinSegments(pathToRoot(fileData.slug!), "static/background", background.name),
  }))

  return (
    <div
      id="private-background-layer"
      data-backgrounds={JSON.stringify(backgrounds)}
      aria-hidden="true"
      hidden
    />
  )
}

PrivateBackground.afterDOMLoaded = `
(() => {
  const STORAGE_KEY = ${JSON.stringify(storageKey)}
  const OPACITY_STORAGE_KEY = ${JSON.stringify(opacityStorageKey)}
  const DEFAULT_OPACITY = ${JSON.stringify(defaultOpacity)}
  const PRIVATE_MODE_CLASS = ${JSON.stringify(privateModeClass)}
  const CHANGE_EVENT_NAME = ${JSON.stringify(changeEventName)}

  const state = window.__quartzPrivateBackground ?? {
    bound: false,
    observer: null,
  }
  window.__quartzPrivateBackground = state

  const getLayer = () => document.getElementById("private-background-layer")

  const safeGetStorage = (key) => {
    try {
      return localStorage.getItem(key)
    } catch {
      return null
    }
  }

  const parseBackgrounds = () => {
    const layer = getLayer()
    if (!layer) return []

    try {
      const raw = layer.dataset.backgrounds ?? "[]"
      const parsed = JSON.parse(raw)
      return Array.isArray(parsed) ? parsed : []
    } catch {
      return []
    }
  }

  const getOpacity = () => {
    const raw = safeGetStorage(OPACITY_STORAGE_KEY)
    const value = Number.parseInt(raw ?? "", 10)
    if (!Number.isFinite(value)) return DEFAULT_OPACITY
    return Math.min(100, Math.max(0, value))
  }

  const hideBackground = (layer) => {
    layer.hidden = true
    layer.style.backgroundImage = ""
  }

  const applyBackground = () => {
    const layer = getLayer()
    if (!layer) return

    if (!document.body.classList.contains(PRIVATE_MODE_CLASS)) {
      hideBackground(layer)
      return
    }

    const selectedName = safeGetStorage(STORAGE_KEY)
    if (!selectedName) {
      hideBackground(layer)
      return
    }

    const selected = parseBackgrounds().find((background) => background.name === selectedName)
    if (!selected?.src) {
      hideBackground(layer)
      return
    }

    layer.style.backgroundImage = 'url("' + selected.src.replaceAll('"', "%22") + '")'
    layer.style.opacity = String(getOpacity() / 100)
    layer.hidden = false
  }

  const bindListenersOnce = () => {
    if (state.bound) return
    state.bound = true

    document.addEventListener("nav", applyBackground)
    document.addEventListener(CHANGE_EVENT_NAME, applyBackground)

    state.observer = new MutationObserver(applyBackground)
    state.observer.observe(document.body, { attributes: true, attributeFilter: ["class"] })
  }

  bindListenersOnce()
  applyBackground()
})()
`

PrivateBackground.css = `
#private-background-layer {
  position: fixed;
  inset: 0;
  z-index: 0;
  width: 100vw;
  height: 100vh;
  background-position: center;
  background-repeat: no-repeat;
  background-size: cover;
  opacity: 0.15;
  pointer-events: none;
  user-select: none;
}

#private-background-layer[hidden] {
  display: none !important;
}
`

export default (() => PrivateBackground) satisfies QuartzComponentConstructor
