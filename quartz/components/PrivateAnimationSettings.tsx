import styles from "./styles/privateAnimationSettings.scss"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import { pathToRoot, joinSegments } from "../util/path"
import { getPrivateCharacters } from "../util/privateCharacters"

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

const PrivateAnimationSettings: QuartzComponent = ({ fileData }: QuartzComponentProps) => {
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
    <section class="private-animation-settings" data-characters={JSON.stringify(characters)}>
      <div class="private-animation-header">
        <h2 class="private-setting-subtitle">Animation</h2>
        <button
          id="private-animation-toggle"
          class="private-animation-switch"
          type="button"
          role="switch"
          aria-checked="false"
        >
          <span class="private-animation-switch-track">
            <span class="private-animation-switch-knob" />
          </span>
          <span id="private-animation-toggle-label" class="private-animation-switch-label">
            off
          </span>
        </button>
      </div>

      <div class="private-setting-row private-animation-character-row">
        <span class="private-setting-label">character:</span>
        <span id="private-animation-current-character" class="private-setting-current">
          none
        </span>
      </div>
      <div id="private-animation-frame-strip" class="private-animation-frame-strip" />

      <div id="private-animation-details" class="private-animation-details" hidden>
        <div class="private-setting-row private-animation-control-row">
          <label class="private-setting-label" for="private-animation-size">
            size:
          </label>
          <input
            id="private-animation-size"
            class="private-animation-slider"
            type="range"
            min="10"
            max="500"
            step="1"
            value={String(defaultSize)}
          />
          <span id="private-animation-size-value" class="private-setting-current">
            {defaultSize}%
          </span>
        </div>

        <div class="private-setting-row private-animation-control-row">
          <label class="private-setting-label" for="private-animation-opacity">
            opacity:
          </label>
          <input
            id="private-animation-opacity"
            class="private-animation-slider"
            type="range"
            min="0"
            max="100"
            step="1"
            value={String(defaultOpacity)}
          />
          <span id="private-animation-opacity-value" class="private-setting-current">
            {defaultOpacity}%
          </span>
        </div>

        <div class="private-setting-row private-animation-control-row">
          <label class="private-setting-label" for="private-animation-fps">
            fps:
          </label>
          <input
            id="private-animation-fps"
            class="private-animation-slider"
            type="range"
            min="1"
            max="30"
            step="1"
            value={String(defaultFps)}
          />
          <span id="private-animation-fps-value" class="private-setting-current">
            {defaultFps}
          </span>
        </div>

        <div class="private-setting-row private-animation-control-row">
          <label class="private-setting-label" for="private-animation-min-guard">
            min guard:
          </label>
          <input
            id="private-animation-min-guard"
            class="private-animation-slider"
            type="range"
            min="0"
            max="5000"
            step="50"
            value={String(defaultMinGuard)}
          />
          <span id="private-animation-min-guard-value" class="private-setting-current">
            {defaultMinGuard}px
          </span>
        </div>
      </div>
    </section>
  )
}

PrivateAnimationSettings.afterDOMLoaded = `
(() => {
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

  const getRoot = () => document.querySelector(".private-animation-settings")
  const getToggle = () => document.getElementById("private-animation-toggle")
  const getToggleLabel = () => document.getElementById("private-animation-toggle-label")
  const getDetails = () => document.getElementById("private-animation-details")
  const getCurrentCharacter = () => document.getElementById("private-animation-current-character")
  const getFrameStrip = () => document.getElementById("private-animation-frame-strip")
  const getSizeSlider = () => document.getElementById("private-animation-size")
  const getSizeValue = () => document.getElementById("private-animation-size-value")
  const getOpacitySlider = () => document.getElementById("private-animation-opacity")
  const getOpacityValue = () => document.getElementById("private-animation-opacity-value")
  const getFpsSlider = () => document.getElementById("private-animation-fps")
  const getFpsValue = () => document.getElementById("private-animation-fps-value")
  const getMinGuardSlider = () => document.getElementById("private-animation-min-guard")
  const getMinGuardValue = () => document.getElementById("private-animation-min-guard-value")

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

  const parseCharacters = () => {
    const root = getRoot()
    if (!(root instanceof HTMLElement)) return []

    try {
      const raw = root.dataset.characters ?? "[]"
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
  const getSelectedCharacterId = () => safeGetStorage(CHARACTER_STORAGE_KEY)
  const getSelectedCharacter = () => {
    const id = getSelectedCharacterId()
    if (!id) return null
    return parseCharacters().find((character) => character.id === id) ?? null
  }
  const getSize = () => clampNumber(Number.parseInt(safeGetStorage(SIZE_STORAGE_KEY) ?? "", 10), DEFAULT_SIZE, 10, 500)
  const getOpacity = () => clampNumber(Number.parseInt(safeGetStorage(OPACITY_STORAGE_KEY) ?? "", 10), DEFAULT_OPACITY, 0, 100)
  const getFps = () => clampNumber(Number.parseInt(safeGetStorage(FPS_STORAGE_KEY) ?? "", 10), DEFAULT_FPS, 1, 30)
  const getMinGuard = () => clampNumber(Number.parseInt(safeGetStorage(MIN_GUARD_STORAGE_KEY) ?? "", 10), DEFAULT_MIN_GUARD, 0, 5000)

  const emitChange = () => {
    document.dispatchEvent(new CustomEvent(CHANGE_EVENT_NAME))
  }

  const updateSwitch = () => {
    const enabled = isEnabled()
    const toggle = getToggle()
    const label = getToggleLabel()
    const details = getDetails()
    const character = getSelectedCharacter()

    if (toggle instanceof HTMLButtonElement) {
      toggle.setAttribute("aria-checked", enabled ? "true" : "false")
      toggle.classList.toggle("active", enabled)
    }

    if (label) label.textContent = enabled ? "on" : "off"
    if (details) details.hidden = !enabled || !character
  }

  const updateSlider = (slider, label, value, suffix = "") => {
    if (slider instanceof HTMLInputElement) {
      slider.value = String(value)
    }

    if (label) {
      label.textContent = String(value) + suffix
    }
  }

  const updateCurrentCharacter = () => {
    const character = getSelectedCharacter()
    const label = getCurrentCharacter()
    const strip = getFrameStrip()

    if (label) label.textContent = character?.name ?? "none"
    if (!strip) return

    strip.textContent = ""
    if (!character) {
      strip.hidden = true
      return
    }

    const frames = character.frames ?? []
    if (frames.length === 0) {
      strip.hidden = false
      const empty = document.createElement("p")
      empty.className = "private-animation-empty"
      empty.textContent = "No animation frames found for this character."
      strip.appendChild(empty)
      return
    }

    strip.hidden = false
    for (const frame of frames) {
      const image = document.createElement("img")
      image.src = frame.src
      image.alt = ""
      image.loading = "lazy"
      image.setAttribute("aria-hidden", "true")
      strip.appendChild(image)
    }
  }

  const updateControls = () => {
    updateSwitch()
    updateCurrentCharacter()
    updateSlider(getSizeSlider(), getSizeValue(), getSize(), "%")
    updateSlider(getOpacitySlider(), getOpacityValue(), getOpacity(), "%")
    updateSlider(getFpsSlider(), getFpsValue(), getFps())
    updateSlider(getMinGuardSlider(), getMinGuardValue(), getMinGuard(), "px")
  }

  const setEnabled = (enabled) => {
    safeSetStorage(ENABLED_STORAGE_KEY, enabled ? "true" : "false")
    updateControls()
    emitChange()
  }

  const setValue = (key, value) => {
    safeSetStorage(key, String(value))
    updateControls()
    emitChange()
  }

  const bindSettings = () => {
    const root = getRoot()
    if (!(root instanceof HTMLElement) || root.dataset.privateAnimationBound === "true") return
    root.dataset.privateAnimationBound = "true"

    getToggle()?.addEventListener("click", () => setEnabled(!isEnabled()))

    getSizeSlider()?.addEventListener("input", (event) => {
      const target = event.target
      if (target instanceof HTMLInputElement) setValue(SIZE_STORAGE_KEY, clampNumber(Number.parseInt(target.value, 10), DEFAULT_SIZE, 10, 500))
    })

    getOpacitySlider()?.addEventListener("input", (event) => {
      const target = event.target
      if (target instanceof HTMLInputElement) setValue(OPACITY_STORAGE_KEY, clampNumber(Number.parseInt(target.value, 10), DEFAULT_OPACITY, 0, 100))
    })

    getFpsSlider()?.addEventListener("input", (event) => {
      const target = event.target
      if (target instanceof HTMLInputElement) setValue(FPS_STORAGE_KEY, clampNumber(Number.parseInt(target.value, 10), DEFAULT_FPS, 1, 30))
    })

    getMinGuardSlider()?.addEventListener("input", (event) => {
      const target = event.target
      if (target instanceof HTMLInputElement) setValue(MIN_GUARD_STORAGE_KEY, clampNumber(Number.parseInt(target.value, 10), DEFAULT_MIN_GUARD, 0, 5000))
    })

    document.addEventListener(CHARACTER_CHANGE_EVENT_NAME, updateControls)
    document.addEventListener(CHANGE_EVENT_NAME, updateControls)
  }

  const init = () => {
    bindSettings()
    updateControls()
  }

  document.addEventListener("nav", init)
  init()
})()
`

PrivateAnimationSettings.css = styles

export default (() => PrivateAnimationSettings) satisfies QuartzComponentConstructor
