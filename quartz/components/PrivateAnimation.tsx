import styles from "./styles/privateAnimation.scss"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import { pathToRoot, joinSegments } from "../util/path"
import { getPrivateCharacters } from "../util/privateCharacters"

const privateModeClass = "private-mode"
const enabledStorageKey = "private-animation-enabled"
const characterStorageKey = "private-character"
const sizeStorageKey = "private-animation-size"
const opacityStorageKey = "private-animation-opacity"
const fpsStorageKey = "private-animation-fps"
const minGuardStorageKey = "private-animation-min-guard"
const changeEventName = "private-animation-change"
const characterChangeEventName = "private-character-change"
const defaultSize = 100
const defaultOpacity = 100
const defaultFps = 8
const defaultMinGuard = 1200
const baseHeightPx = 64

const PrivateAnimation: QuartzComponent = ({ fileData }: QuartzComponentProps) => {
  const characters = getPrivateCharacters().map((character) => ({
    id: character.id,
    name: character.name,
    frames: character.animation.frames.map((frame) => ({
      name: frame.name,
      path: frame.path,
      src: joinSegments(pathToRoot(fileData.slug!), "static/characters", character.id, frame.path),
    })),
  }))

  return (
    <div id="private-animation-layer" data-characters={JSON.stringify(characters)} aria-hidden="true" hidden>
      <img id="private-animation-sprite" alt="" aria-hidden="true" />
      <div id="private-animation-progress" aria-hidden="true">
        <div id="private-animation-progress-fill" />
        <input
          id="private-animation-progress-range"
          type="range"
          min="0"
          max="100000"
          step="1"
          defaultValue="0"
          tabIndex={-1}
          aria-hidden="true"
        />
      </div>
    </div>
  )
}

PrivateAnimation.afterDOMLoaded = `
(() => {
  const PRIVATE_MODE_CLASS = ${JSON.stringify(privateModeClass)}
  const ANIMATION_ACTIVE_CLASS = "private-animation-active"
  const ENABLED_STORAGE_KEY = ${JSON.stringify(enabledStorageKey)}
  const CHARACTER_STORAGE_KEY = ${JSON.stringify(characterStorageKey)}
  const SIZE_STORAGE_KEY = ${JSON.stringify(sizeStorageKey)}
  const OPACITY_STORAGE_KEY = ${JSON.stringify(opacityStorageKey)}
  const FPS_STORAGE_KEY = ${JSON.stringify(fpsStorageKey)}
  const MIN_GUARD_STORAGE_KEY = ${JSON.stringify(minGuardStorageKey)}
  const CHANGE_EVENT_NAME = ${JSON.stringify(changeEventName)}
  const CHARACTER_CHANGE_EVENT_NAME = ${JSON.stringify(characterChangeEventName)}
  const DEFAULT_SIZE = ${JSON.stringify(defaultSize)}
  const DEFAULT_OPACITY = ${JSON.stringify(defaultOpacity)}
  const DEFAULT_FPS = ${JSON.stringify(defaultFps)}
  const DEFAULT_MIN_GUARD = ${JSON.stringify(defaultMinGuard)}
  const BASE_HEIGHT_PX = ${JSON.stringify(baseHeightPx)}
  const RANGE_MAX = 100000

  const state = window.__quartzPrivateAnimation ?? {
    bound: false,
    observer: null,
    frameTimer: null,
    frameIndex: 0,
    currentCharacter: null,
    currentFps: null,
    raf: 0,
  }
  window.__quartzPrivateAnimation = state

  const getLayer = () => document.getElementById("private-animation-layer")
  const getSprite = () => document.getElementById("private-animation-sprite")
  const getProgress = () => document.getElementById("private-animation-progress")
  const getProgressFill = () => document.getElementById("private-animation-progress-fill")
  const getProgressRange = () => document.getElementById("private-animation-progress-range")

  const safeGetStorage = (key) => {
    try {
      return localStorage.getItem(key)
    } catch {
      return null
    }
  }

  const parseCharacters = () => {
    const layer = getLayer()
    if (!(layer instanceof HTMLElement)) return []

    try {
      const raw = layer.dataset.characters ?? "[]"
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

  const getEnabled = () => safeGetStorage(ENABLED_STORAGE_KEY) === "true"
  const getSelectedCharacterId = () => safeGetStorage(CHARACTER_STORAGE_KEY)
  const clampRatio = (ratio) => Math.min(1, Math.max(0, ratio))
  const getSize = () => clampNumber(Number.parseInt(safeGetStorage(SIZE_STORAGE_KEY) ?? "", 10), DEFAULT_SIZE, 10, 500)
  const getOpacity = () => clampNumber(Number.parseInt(safeGetStorage(OPACITY_STORAGE_KEY) ?? "", 10), DEFAULT_OPACITY, 0, 100)
  const getFps = () => clampNumber(Number.parseInt(safeGetStorage(FPS_STORAGE_KEY) ?? "", 10), DEFAULT_FPS, 1, 30)
  const getMinGuard = () => clampNumber(Number.parseInt(safeGetStorage(MIN_GUARD_STORAGE_KEY) ?? "", 10), DEFAULT_MIN_GUARD, 0, 10000)

  const getSelectedCharacter = () => {
    const id = getSelectedCharacterId()
    if (!id) return null
    return parseCharacters().find((character) => character.id === id) ?? null
  }

  const stopFrameTimer = () => {
    if (state.frameTimer) {
      window.clearInterval(state.frameTimer)
      state.frameTimer = null
    }
  }

  const setAnimationActive = (active) => {
    document.body.classList.toggle(ANIMATION_ACTIVE_CLASS, active)
    document.documentElement.classList.toggle(ANIMATION_ACTIVE_CLASS, active)

    if (active) {
      document.documentElement.style.setProperty("scrollbar-width", "none")
      document.documentElement.style.setProperty("-ms-overflow-style", "none")
      document.body.style.setProperty("scrollbar-width", "none")
      document.body.style.setProperty("-ms-overflow-style", "none")
    } else {
      document.documentElement.style.removeProperty("scrollbar-width")
      document.documentElement.style.removeProperty("-ms-overflow-style")
      document.body.style.removeProperty("scrollbar-width")
      document.body.style.removeProperty("-ms-overflow-style")
    }
  }

  const hideAnimation = () => {
    const layer = getLayer()
    const sprite = getSprite()
    const fill = getProgressFill()

    stopFrameTimer()
    state.currentCharacter = null
    state.currentFps = null
    state.frameIndex = 0

    setAnimationActive(false)

    if (layer) layer.hidden = true
    if (fill instanceof HTMLElement) fill.style.width = "0%"
    if (sprite instanceof HTMLImageElement) sprite.removeAttribute("src")
  }

  const getScrollRoot = () => document.scrollingElement || document.documentElement

  const getScrollableHeight = () => {
    const root = getScrollRoot()
    const scrollHeight = Math.max(
      root.scrollHeight,
      document.documentElement.scrollHeight,
      document.body.scrollHeight,
    )

    return Math.max(0, scrollHeight - window.innerHeight)
  }

  const getScrollRatio = () => {
    const scrollable = getScrollableHeight()

    if (scrollable <= 0) return 1

    const root = getScrollRoot()
    const scrollTop = Math.max(
      window.scrollY,
      root.scrollTop,
      document.documentElement.scrollTop,
      document.body.scrollTop,
      0,
    )

    return Math.min(1, Math.max(0, scrollTop / scrollable))
  }

  const updateSpritePositionFromRatio = (ratio) => {
    const sprite = getSprite()
    const progress = getProgress()

    if (!(sprite instanceof HTMLImageElement) || !sprite.src) return

    const spriteWidth = getSpriteVisualWidth()
    const safeRatio = clampRatio(ratio)

    if (progress instanceof HTMLElement) {
      const rect = progress.getBoundingClientRect()

      if (rect.width > 0) {
        const centerX = rect.left + rect.width * safeRatio
        sprite.style.left = Math.round(centerX - spriteWidth / 2) + "px"
        return
      }
    }

    const maxLeft = Math.max(0, window.innerWidth - spriteWidth)
    sprite.style.left = Math.round(maxLeft * safeRatio) + "px"
  }

  const updateProgressBar = (ratio = getScrollRatio()) => {
    const safeRatio = clampRatio(ratio)

    const fill = getProgressFill()
    if (fill instanceof HTMLElement) {
      fill.style.width = Math.round(safeRatio * 10000) / 100 + "%"
    }

    const range = getProgressRange()
    if (range instanceof HTMLInputElement) {
      range.value = String(Math.round(safeRatio * RANGE_MAX))
    }
  }

  const getSpriteVisualWidth = () => {
    const sprite = getSprite()
    if (!(sprite instanceof HTMLImageElement)) return BASE_HEIGHT_PX

    const rect = sprite.getBoundingClientRect()
    if (rect.width > 0) return rect.width

    const height = Number.parseFloat(sprite.style.height)
    if (
      Number.isFinite(height) &&
      height > 0 &&
      sprite.naturalWidth > 0 &&
      sprite.naturalHeight > 0
    ) {
      return (height * sprite.naturalWidth) / sprite.naturalHeight
    }

    return Math.max(1, Math.round((BASE_HEIGHT_PX * getSize()) / 100))
  }

  const scrollToRatio = (ratio) => {
    const target = Math.round(getScrollableHeight() * Math.min(1, Math.max(0, ratio)))

    window.scrollTo({
      top: target,
      behavior: "auto",
    })
  }

  const updatePosition = () => {
    const ratio = getScrollRatio()
    updateSpritePositionFromRatio(ratio)
    updateProgressBar(ratio)
  }
  const schedulePositionUpdate = () => {
    if (state.raf) return
    state.raf = window.requestAnimationFrame(() => {
      state.raf = 0
      updatePosition()
    })
  }

  const applySpriteStyle = () => {
    const sprite = getSprite()
    if (!(sprite instanceof HTMLImageElement)) return

    sprite.style.height = Math.max(1, Math.round((BASE_HEIGHT_PX * getSize()) / 100)) + "px"
    sprite.style.opacity = String(getOpacity() / 100)

    // Keep this sprite independent from Quartz's global article image styles.
    sprite.style.setProperty("position", "fixed", "important")
    sprite.style.setProperty("bottom", "23px", "important")
    sprite.style.setProperty("margin", "0", "important")
    sprite.style.setProperty("padding", "0", "important")
    sprite.style.setProperty("border", "0", "important")
    sprite.style.setProperty("border-radius", "0", "important")
    sprite.style.setProperty("max-width", "none", "important")
    sprite.style.setProperty("max-height", "none", "important")
    sprite.style.setProperty("transform", "none", "important")

    schedulePositionUpdate()
  }

  const setFrame = (character, frameIndex) => {
    const sprite = getSprite()
    if (!(sprite instanceof HTMLImageElement)) return

    const frames = character.frames ?? []
    if (frames.length === 0) return

    const frame = frames[frameIndex % frames.length]
    if (!frame?.src) return

    if (sprite.src !== new URL(frame.src, window.location.href).href) {
      sprite.src = frame.src
    }
    applySpriteStyle()
  }

  const startFrameTimer = (character) => {
    stopFrameTimer()

    const fps = getFps()
    state.currentCharacter = character.id
    state.currentFps = fps
    state.frameIndex = 0
    setFrame(character, state.frameIndex)

    const frames = character.frames ?? []
    if (frames.length <= 1) return

    state.frameTimer = window.setInterval(() => {
      state.frameIndex = (state.frameIndex + 1) % frames.length
      setFrame(character, state.frameIndex)
    }, Math.max(33, Math.round(1000 / fps)))
  }

  const applyAnimation = () => {
    const layer = getLayer()
    if (!(layer instanceof HTMLElement)) return

    if (!document.body.classList.contains(PRIVATE_MODE_CLASS) || !getEnabled()) {
      hideAnimation()
      return
    }

    const character = getSelectedCharacter()
    if (!character || !Array.isArray(character.frames) || character.frames.length === 0) {
      hideAnimation()
      return
    }

    layer.hidden = false
    setAnimationActive(true)
    applySpriteStyle()
    updateProgressBar()

    const fps = getFps()

    if (state.currentCharacter !== character.id || state.currentFps !== fps || !state.frameTimer) {
      startFrameTimer(character)
    } else {
      setFrame(character, state.frameIndex)
    }
  }

  const bindListenersOnce = () => {
    if (state.bound) return
    state.bound = true

    document.addEventListener("nav", applyAnimation)
    document.addEventListener(CHANGE_EVENT_NAME, applyAnimation)
    document.addEventListener(CHARACTER_CHANGE_EVENT_NAME, applyAnimation)
    window.addEventListener("scroll", schedulePositionUpdate, { passive: true })
    window.addEventListener("resize", schedulePositionUpdate)

    const progress = getProgress()
    const range = getProgressRange()

    if (progress instanceof HTMLElement && range instanceof HTMLInputElement) {
      const syncFromRange = () => {
        const ratio = clampRatio(Number(range.value) / RANGE_MAX)

        scrollToRatio(ratio)
        updateSpritePositionFromRatio(ratio)
        updateProgressBar(ratio)
      }

      const stopRangeDrag = () => {
        progress.classList.remove("is-dragging")
      }

      range.addEventListener("input", syncFromRange)

      range.addEventListener("pointerdown", () => {
        progress.classList.add("is-dragging")
      })

      range.addEventListener("pointerup", stopRangeDrag)
      range.addEventListener("pointercancel", stopRangeDrag)
      range.addEventListener("blur", stopRangeDrag)
      document.addEventListener("pointerup", stopRangeDrag)
      document.addEventListener("pointercancel", stopRangeDrag)
    }

    const sprite = getSprite()
    if (sprite instanceof HTMLImageElement) {
      sprite.addEventListener("load", schedulePositionUpdate)
    }

    state.observer = new MutationObserver(applyAnimation)
    state.observer.observe(document.body, { attributes: true, attributeFilter: ["class"] })
  }

  bindListenersOnce()
  applyAnimation()
})()
`

PrivateAnimation.css = styles

export default (() => PrivateAnimation) satisfies QuartzComponentConstructor
