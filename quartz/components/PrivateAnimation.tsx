import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import { pathToRoot, joinSegments } from "../util/path"
import { getPrivateAnimationCharacters } from "../util/privateAnimations"

const privateModeClass = "private-mode"
const enabledStorageKey = "private-animation-enabled"
const characterStorageKey = "private-animation-character"
const sizeStorageKey = "private-animation-size"
const opacityStorageKey = "private-animation-opacity"
const fpsStorageKey = "private-animation-fps"
const minGuardStorageKey = "private-animation-min-guard"
const changeEventName = "private-animation-change"
const defaultSize = 100
const defaultOpacity = 100
const defaultFps = 8
const defaultMinGuard = 1200
const baseHeightPx = 64

const PrivateAnimation: QuartzComponent = ({ fileData }: QuartzComponentProps) => {
  const characters = getPrivateAnimationCharacters().map((character) => ({
    name: character.name,
    frames: character.frames.map((frame) => ({
      name: frame.name,
      src: joinSegments(pathToRoot(fileData.slug!), "static/animation", character.name, frame.name),
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
  const getSelectedCharacterName = () => safeGetStorage(CHARACTER_STORAGE_KEY)
  const clampRatio = (ratio) => Math.min(1, Math.max(0, ratio))
  const getSize = () => clampNumber(Number.parseInt(safeGetStorage(SIZE_STORAGE_KEY) ?? "", 10), DEFAULT_SIZE, 10, 500)
  const getOpacity = () => clampNumber(Number.parseInt(safeGetStorage(OPACITY_STORAGE_KEY) ?? "", 10), DEFAULT_OPACITY, 0, 100)
  const getFps = () => clampNumber(Number.parseInt(safeGetStorage(FPS_STORAGE_KEY) ?? "", 10), DEFAULT_FPS, 1, 30)
  const getMinGuard = () => clampNumber(Number.parseInt(safeGetStorage(MIN_GUARD_STORAGE_KEY) ?? "", 10), DEFAULT_MIN_GUARD, 0, 10000)

  const getSelectedCharacter = () => {
    const name = getSelectedCharacterName()
    if (!name) return null
    return parseCharacters().find((character) => character.name === name) ?? null
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
    state.currentCharacter = character.name
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

    if (state.currentCharacter !== character.name || state.currentFps !== fps || !state.frameTimer) {
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

PrivateAnimation.css = `
#private-animation-layer {
  position: fixed;
  left: 0;
  right: 0;
  bottom: 0;
  z-index: 9998;
  height: 0;
  pointer-events: none;
  user-select: none;
  overflow: visible;
}

#private-animation-layer[hidden] {
  display: none !important;
}

#private-animation-sprite {
  position: fixed !important;
  left: 0;
  bottom: 6px !important;
  display: block !important;
  box-sizing: border-box;
  width: auto !important;
  height: auto;
  max-width: none !important;
  max-height: none !important;
  margin: 0 !important;
  padding: 0 !important;
  border: 0 !important;
  border-radius: 0 !important;
  object-fit: contain;
  pointer-events: none;
  user-select: none;
  image-rendering: auto;
  transform: none !important;
  transform-origin: bottom left;
}

html.private-animation-active,
body.private-animation-active {
  scrollbar-width: none !important;
  -ms-overflow-style: none !important;
}

html.private-animation-active::-webkit-scrollbar,
body.private-animation-active::-webkit-scrollbar {
  display: none !important;
  width: 0 !important;
  height: 0 !important;
}

#private-animation-progress {
  position: fixed;
  left: max(16px, env(safe-area-inset-left));
  right: max(16px, env(safe-area-inset-right));
  bottom: 8px;
  height: 7px;
  z-index: 10000;
  box-sizing: border-box;
  pointer-events: auto;
  cursor: pointer;
  touch-action: none;
  user-select: none;
  overflow: visible;
  border-radius: 999px;
  background: color-mix(in srgb, var(--lightgray) 82%, var(--light) 18%);
  border: 1px solid color-mix(in srgb, var(--darkgray) 18%, transparent);
  box-shadow: 0 2px 10px color-mix(in srgb, var(--darkgray) 16%, transparent);
}

#private-animation-progress-fill {
  position: absolute;
  inset: 0 auto 0 0;
  height: 100%;
  width: 0%;
  z-index: 1;
  border-radius: inherit;
  pointer-events: none;
  background: linear-gradient(90deg, var(--secondary), var(--tertiary));
  opacity: 0.9;
  transition: width 80ms linear;
}

#private-animation-progress-range {
  position: absolute;
  left: 0;
  right: 0;
  top: 50%;
  width: 100%;
  height: 28px;
  z-index: 2;
  margin: 0;
  padding: 0;
  opacity: 0;
  cursor: pointer;
  transform: translateY(-50%);
  appearance: none;
  -webkit-appearance: none;
  background: transparent;
  touch-action: none;
}

#private-animation-progress-range::-webkit-slider-runnable-track {
  height: 28px;
  background: transparent;
  border: 0;
}

#private-animation-progress-range::-webkit-slider-thumb {
  width: 28px;
  height: 28px;
  border: 0;
  border-radius: 999px;
  background: transparent;
  appearance: none;
  -webkit-appearance: none;
}

#private-animation-progress-range::-moz-range-track {
  height: 28px;
  background: transparent;
  border: 0;
}

#private-animation-progress-range::-moz-range-thumb {
  width: 28px;
  height: 28px;
  border: 0;
  border-radius: 999px;
  background: transparent;
}

#private-animation-progress.is-dragging #private-animation-progress-fill {
  transition: none;
}
`

export default (() => PrivateAnimation) satisfies QuartzComponentConstructor
