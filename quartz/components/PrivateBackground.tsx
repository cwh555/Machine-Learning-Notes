import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import { pathToRoot, joinSegments } from "../util/path"
import { getPrivateBackgroundFiles } from "../util/privateBackgrounds"

const enabledStorageKey = "private-background-enabled"
const storageKey = "private-background"
const opacityStorageKey = "private-background-opacity"
const slideshowEnabledStorageKey = "private-background-slideshow-enabled"
const slideshowSecondsStorageKey = "private-background-slideshow-seconds"
const defaultOpacity = 15
const defaultSlideshowSeconds = 8
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
  const ENABLED_STORAGE_KEY = ${JSON.stringify(enabledStorageKey)}
  const STORAGE_KEY = ${JSON.stringify(storageKey)}
  const OPACITY_STORAGE_KEY = ${JSON.stringify(opacityStorageKey)}
  const SLIDESHOW_ENABLED_STORAGE_KEY = ${JSON.stringify(slideshowEnabledStorageKey)}
  const SLIDESHOW_SECONDS_STORAGE_KEY = ${JSON.stringify(slideshowSecondsStorageKey)}
  const DEFAULT_OPACITY = ${JSON.stringify(defaultOpacity)}
  const DEFAULT_SLIDESHOW_SECONDS = ${JSON.stringify(defaultSlideshowSeconds)}
  const PRIVATE_MODE_CLASS = ${JSON.stringify(privateModeClass)}
  const CHANGE_EVENT_NAME = ${JSON.stringify(changeEventName)}

  const state = window.__quartzPrivateBackground ?? {
    bound: false,
    observer: null,
    slideshowTimer: null,
    slideshowMs: null,
    slideshowSignature: null,
    shuffleQueue: [],
    lastBackgroundName: null,
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

  const safeSetStorage = (key, value) => {
    try {
      localStorage.setItem(key, value)
    } catch {}
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

  const clampNumber = (value, fallback, min, max) => {
    if (!Number.isFinite(value)) return fallback
    return Math.min(max, Math.max(min, Math.round(value)))
  }

  const isEnabled = () => safeGetStorage(ENABLED_STORAGE_KEY) === "true"
  const isSlideshowEnabled = () => safeGetStorage(SLIDESHOW_ENABLED_STORAGE_KEY) === "true"

  const getOpacity = () => {
    const raw = safeGetStorage(OPACITY_STORAGE_KEY)
    return clampNumber(Number.parseInt(raw ?? "", 10), DEFAULT_OPACITY, 0, 100)
  }

  const getSlideshowSeconds = () => {
    const raw = safeGetStorage(SLIDESHOW_SECONDS_STORAGE_KEY)
    return clampNumber(Number.parseInt(raw ?? "", 10), DEFAULT_SLIDESHOW_SECONDS, 1, 3600)
  }

  const getBackgroundSignature = (backgrounds) =>
    JSON.stringify(backgrounds.map((background) => [background.name, background.src]))

  const stopSlideshowTimer = () => {
    if (state.slideshowTimer) {
      window.clearInterval(state.slideshowTimer)
      state.slideshowTimer = null
    }

    state.slideshowMs = null
    state.slideshowSignature = null
  }

  const hideBackground = (layer) => {
    stopSlideshowTimer()
    layer.hidden = true
    layer.style.backgroundImage = ""
  }

  const applyLayerBackground = (layer, background) => {
    layer.style.backgroundImage = 'url("' + background.src.replaceAll('"', "%22") + '")'
    layer.style.opacity = String(getOpacity() / 100)
    layer.hidden = false
  }

  const shuffleNames = (names) => {
    const shuffled = [...names]

    for (let index = shuffled.length - 1; index > 0; index--) {
      const swapIndex = Math.floor(Math.random() * (index + 1))
      const temporary = shuffled[index]
      shuffled[index] = shuffled[swapIndex]
      shuffled[swapIndex] = temporary
    }

    if (shuffled.length > 1 && shuffled[0] === state.lastBackgroundName) {
      const temporary = shuffled[0]
      shuffled[0] = shuffled[1]
      shuffled[1] = temporary
    }

    return shuffled
  }

  const refillShuffleQueue = (backgrounds) => {
    state.shuffleQueue = shuffleNames(backgrounds.map((background) => background.name))
  }

  const normalizeShuffleQueue = (backgrounds, currentName) => {
    const validNames = new Set(backgrounds.map((background) => background.name))
    state.shuffleQueue = Array.isArray(state.shuffleQueue)
      ? state.shuffleQueue.filter((name) => validNames.has(name))
      : []

    if (state.shuffleQueue.length === 0) {
      refillShuffleQueue(backgrounds)
    }

    if (currentName && validNames.has(currentName)) {
      state.lastBackgroundName = currentName
      state.shuffleQueue = state.shuffleQueue.filter((name) => name !== currentName)
    }
  }

  const takeNextRandomBackground = (backgrounds) => {
    if (backgrounds.length === 0) return null

    const validNames = new Set(backgrounds.map((background) => background.name))
    state.shuffleQueue = Array.isArray(state.shuffleQueue)
      ? state.shuffleQueue.filter((name) => validNames.has(name))
      : []

    if (state.shuffleQueue.length === 0) {
      refillShuffleQueue(backgrounds)
    }

    const nextName = state.shuffleQueue.shift()
    const nextBackground = backgrounds.find((background) => background.name === nextName) ?? backgrounds[0]
    state.lastBackgroundName = nextBackground.name
    return nextBackground
  }

  const ensureSlideshowTimer = (layer, backgrounds) => {
    const seconds = getSlideshowSeconds()
    const milliseconds = Math.max(1000, seconds * 1000)
    const signature = getBackgroundSignature(backgrounds)

    if (
      state.slideshowTimer &&
      state.slideshowMs === milliseconds &&
      state.slideshowSignature === signature
    ) {
      return
    }

    stopSlideshowTimer()
    state.slideshowMs = milliseconds
    state.slideshowSignature = signature

    state.slideshowTimer = window.setInterval(() => {
      const nextBackground = takeNextRandomBackground(backgrounds)
      if (!nextBackground) return

      safeSetStorage(STORAGE_KEY, nextBackground.name)
      applyLayerBackground(layer, nextBackground)
      document.dispatchEvent(
        new CustomEvent(CHANGE_EVENT_NAME, {
          detail: { name: nextBackground.name, source: "slideshow" },
        }),
      )
    }, milliseconds)
  }

  const applyBackground = () => {
    const layer = getLayer()
    if (!layer) return

    if (!document.body.classList.contains(PRIVATE_MODE_CLASS) || !isEnabled()) {
      hideBackground(layer)
      return
    }

    const backgrounds = parseBackgrounds()
    if (backgrounds.length === 0) {
      hideBackground(layer)
      return
    }

    const selectedName = safeGetStorage(STORAGE_KEY)
    let selected = backgrounds.find((background) => background.name === selectedName)

    if (isSlideshowEnabled()) {
      if (!selected) {
        selected = takeNextRandomBackground(backgrounds)
        if (selected) safeSetStorage(STORAGE_KEY, selected.name)
      }

      if (selected) {
        normalizeShuffleQueue(backgrounds, selected.name)
        applyLayerBackground(layer, selected)
        ensureSlideshowTimer(layer, backgrounds)
        return
      }
    }

    stopSlideshowTimer()

    if (!selected?.src) {
      hideBackground(layer)
      return
    }

    state.lastBackgroundName = selected.name
    applyLayerBackground(layer, selected)
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
