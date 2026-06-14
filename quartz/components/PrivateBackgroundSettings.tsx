import styles from "./styles/privateBackgroundSettings.scss"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import { pathToRoot, joinSegments } from "../util/path"
import { getPrivateCharacters } from "../util/privateCharacters"

const characterStorageKey = "private-character"
const enabledStorageKey = "private-background-enabled"
const storageKey = "private-background"
const opacityStorageKey = "private-background-opacity"
const slideshowEnabledStorageKey = "private-background-slideshow-enabled"
const slideshowSecondsStorageKey = "private-background-slideshow-seconds"
const defaultOpacity = 15
const defaultSlideshowSeconds = 8
const changeEventName = "private-background-change"
const characterChangeEventName = "private-character-change"

const PrivateBackgroundSettings: QuartzComponent = ({ fileData }: QuartzComponentProps) => {
  const characters = getPrivateCharacters().map((character) => ({
    id: character.id,
    name: character.name,
    backgrounds: character.backgrounds.map((background) => ({
      name: background.name,
      path: background.path,
      src: joinSegments(pathToRoot(fileData.slug!), "static/characters", character.id, background.path),
    })),
  }))

  return (
    <section class="private-background-settings" data-characters={JSON.stringify(characters)}>
      <div class="private-background-header">
        <h2 class="private-setting-subtitle">Background</h2>
        <button
          id="private-background-toggle"
          class="private-background-switch"
          type="button"
          role="switch"
          aria-checked="false"
        >
          <span class="private-background-switch-track">
            <span class="private-background-switch-knob" />
          </span>
          <span id="private-background-toggle-label" class="private-background-switch-label">
            off
          </span>
        </button>
      </div>

      <div class="private-setting-row private-background-character-row">
        <span class="private-setting-label">character:</span>
        <span id="private-background-character" class="private-setting-current">
          none
        </span>
      </div>

      <div id="private-background-details" class="private-background-details" hidden>
        <div class="private-setting-row">
          <span class="private-setting-label">current:</span>
          <span id="private-background-current" class="private-setting-current">
            none
          </span>
          <button id="private-background-select" class="private-setting-button" type="button">
            select new
          </button>
        </div>

        <div class="private-setting-row private-opacity-row">
          <label class="private-setting-label" for="private-background-opacity">
            opacity:
          </label>
          <input
            id="private-background-opacity"
            class="private-opacity-slider"
            type="range"
            min="0"
            max="100"
            step="1"
            value={String(defaultOpacity)}
          />
          <span id="private-background-opacity-value" class="private-setting-current">
            {defaultOpacity}%
          </span>
        </div>

        <div class="private-background-slideshow-block">
          <div class="private-setting-row private-background-slideshow-row">
            <span class="private-setting-label">random cycle:</span>
            <button
              id="private-background-slideshow-toggle"
              class="private-background-switch"
              type="button"
              role="switch"
              aria-checked="false"
            >
              <span class="private-background-switch-track">
                <span class="private-background-switch-knob" />
              </span>
              <span id="private-background-slideshow-toggle-label" class="private-background-switch-label">
                off
              </span>
            </button>
          </div>

          <div id="private-background-slideshow-details" class="private-background-slideshow-details" hidden>
            <div class="private-setting-row private-background-seconds-row">
              <label class="private-setting-label" for="private-background-slideshow-seconds">
                interval:
              </label>
              <input
                id="private-background-slideshow-seconds"
                class="private-background-seconds-input"
                type="number"
                min="1"
                max="3600"
                step="1"
                value={String(defaultSlideshowSeconds)}
              />
              <span id="private-background-slideshow-seconds-value" class="private-setting-current">
                {defaultSlideshowSeconds}s
              </span>
            </div>
          </div>
        </div>
      </div>

      <div id="private-background-modal" class="private-background-modal" hidden>
        <div class="private-background-panel" role="dialog" aria-modal="true" aria-label="Select background">
          <div class="private-background-panel-header">
            <h2>Background</h2>
            <button id="private-background-close" class="private-background-close" type="button" aria-label="Close">
              ×
            </button>
          </div>
          <div id="private-background-grid" class="private-background-grid" />
        </div>
      </div>
    </section>
  )
}

PrivateBackgroundSettings.afterDOMLoaded = `
(() => {
  const CHARACTER_STORAGE_KEY = ${JSON.stringify(characterStorageKey)}
  const ENABLED_STORAGE_KEY = ${JSON.stringify(enabledStorageKey)}
  const STORAGE_KEY = ${JSON.stringify(storageKey)}
  const OPACITY_STORAGE_KEY = ${JSON.stringify(opacityStorageKey)}
  const SLIDESHOW_ENABLED_STORAGE_KEY = ${JSON.stringify(slideshowEnabledStorageKey)}
  const SLIDESHOW_SECONDS_STORAGE_KEY = ${JSON.stringify(slideshowSecondsStorageKey)}
  const DEFAULT_OPACITY = ${JSON.stringify(defaultOpacity)}
  const DEFAULT_SLIDESHOW_SECONDS = ${JSON.stringify(defaultSlideshowSeconds)}
  const CHANGE_EVENT_NAME = ${JSON.stringify(changeEventName)}
  const CHARACTER_CHANGE_EVENT_NAME = ${JSON.stringify(characterChangeEventName)}

  const getRoot = () => document.querySelector(".private-background-settings")
  const getToggle = () => document.getElementById("private-background-toggle")
  const getToggleLabel = () => document.getElementById("private-background-toggle-label")
  const getCharacterLabel = () => document.getElementById("private-background-character")
  const getDetails = () => document.getElementById("private-background-details")
  const getCurrent = () => document.getElementById("private-background-current")
  const getSelectButton = () => document.getElementById("private-background-select")
  const getOpacitySlider = () => document.getElementById("private-background-opacity")
  const getOpacityValue = () => document.getElementById("private-background-opacity-value")
  const getSlideshowToggle = () => document.getElementById("private-background-slideshow-toggle")
  const getSlideshowToggleLabel = () => document.getElementById("private-background-slideshow-toggle-label")
  const getSlideshowDetails = () => document.getElementById("private-background-slideshow-details")
  const getSlideshowSecondsInput = () => document.getElementById("private-background-slideshow-seconds")
  const getSlideshowSecondsValue = () => document.getElementById("private-background-slideshow-seconds-value")
  const getModal = () => document.getElementById("private-background-modal")
  const getGrid = () => document.getElementById("private-background-grid")
  const getCloseButton = () => document.getElementById("private-background-close")

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

  const clampNumber = (value, fallback, min, max) => {
    if (!Number.isFinite(value)) return fallback
    return Math.min(max, Math.max(min, Math.round(value)))
  }

  const clampOpacity = (value) => clampNumber(value, DEFAULT_OPACITY, 0, 100)
  const clampSeconds = (value) => clampNumber(value, DEFAULT_SLIDESHOW_SECONDS, 1, 3600)

  const isEnabled = () => safeGetStorage(ENABLED_STORAGE_KEY) === "true"
  const isSlideshowEnabled = () => safeGetStorage(SLIDESHOW_ENABLED_STORAGE_KEY) === "true"

  const getOpacity = () => {
    const raw = safeGetStorage(OPACITY_STORAGE_KEY)
    return clampOpacity(Number.parseInt(raw ?? "", 10))
  }

  const getSlideshowSeconds = () => {
    const raw = safeGetStorage(SLIDESHOW_SECONDS_STORAGE_KEY)
    return clampSeconds(Number.parseInt(raw ?? "", 10))
  }

  const emitChange = (detail = {}) => {
    document.dispatchEvent(new CustomEvent(CHANGE_EVENT_NAME, { detail }))
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

  const getSelectedCharacter = () => {
    const id = safeGetStorage(CHARACTER_STORAGE_KEY)
    if (!id) return null
    return parseCharacters().find((character) => character.id === id) ?? null
  }

  const getBackgrounds = () => getSelectedCharacter()?.backgrounds ?? []
  const getSelectedPath = () => safeGetStorage(STORAGE_KEY)

  const updateCurrentLabel = () => {
    const current = getCurrent()
    const characterLabel = getCharacterLabel()
    const character = getSelectedCharacter()

    if (characterLabel) characterLabel.textContent = character?.name ?? "none"
    if (!current) return

    const selectedPath = getSelectedPath()
    const selected = getBackgrounds().find((background) => background.path === selectedPath)
    current.textContent = selected?.name ?? "none"
  }

  const updateSwitchButton = (toggle, label, enabled) => {
    if (toggle instanceof HTMLButtonElement) {
      toggle.setAttribute("aria-checked", enabled ? "true" : "false")
      toggle.classList.toggle("active", enabled)
    }

    if (label) label.textContent = enabled ? "on" : "off"
  }

  const updateControls = () => {
    const enabled = isEnabled()
    const slideshowEnabled = isSlideshowEnabled()
    const details = getDetails()
    const slideshowDetails = getSlideshowDetails()
    const opacity = getOpacity()
    const seconds = getSlideshowSeconds()
    const opacitySlider = getOpacitySlider()
    const opacityValue = getOpacityValue()
    const secondsInput = getSlideshowSecondsInput()
    const secondsValue = getSlideshowSecondsValue()
    const character = getSelectedCharacter()

    updateSwitchButton(getToggle(), getToggleLabel(), enabled)
    updateSwitchButton(getSlideshowToggle(), getSlideshowToggleLabel(), slideshowEnabled)

    if (details) details.hidden = !enabled || !character
    if (slideshowDetails) slideshowDetails.hidden = !enabled || !slideshowEnabled || !character

    if (opacitySlider instanceof HTMLInputElement) opacitySlider.value = String(opacity)
    if (opacityValue) opacityValue.textContent = opacity + "%"

    if (secondsInput instanceof HTMLInputElement) secondsInput.value = String(seconds)
    if (secondsValue) secondsValue.textContent = seconds + "s"
  }

  const closeModal = () => {
    const modal = getModal()
    if (modal) modal.hidden = true
  }

  const openModal = () => {
    renderGrid()
    const modal = getModal()
    if (modal) modal.hidden = false
  }

  const setEnabled = (enabled) => {
    safeSetStorage(ENABLED_STORAGE_KEY, enabled ? "true" : "false")
    updateControls()
    emitChange({ enabled })
  }

  const setSlideshowEnabled = (enabled) => {
    safeSetStorage(SLIDESHOW_ENABLED_STORAGE_KEY, enabled ? "true" : "false")
    updateControls()
    emitChange({ slideshowEnabled: enabled })
  }

  const setOpacity = (value) => {
    const opacity = clampOpacity(value)
    safeSetStorage(OPACITY_STORAGE_KEY, String(opacity))
    updateControls()
    emitChange({ opacity })
  }

  const setSlideshowSeconds = (value) => {
    const seconds = clampSeconds(value)
    safeSetStorage(SLIDESHOW_SECONDS_STORAGE_KEY, String(seconds))
    updateControls()
    emitChange({ slideshowSeconds: seconds })
  }

  const selectBackground = (background) => {
    safeSetStorage(STORAGE_KEY, background.path)
    updateCurrentLabel()
    renderGrid()
    emitChange({ path: background.path, name: background.name })
    closeModal()
  }

  const renderEmptyState = (grid) => {
    const empty = document.createElement("p")
    empty.className = "private-background-empty"
    empty.textContent = getSelectedCharacter() ? "No background images found for this character." : "Select a character first."
    grid.appendChild(empty)
  }

  const renderGrid = () => {
    const grid = getGrid()
    if (!grid) return

    grid.textContent = ""
    const backgrounds = getBackgrounds()
    if (backgrounds.length === 0) {
      renderEmptyState(grid)
      return
    }

    const selectedPath = getSelectedPath()
    for (const background of backgrounds) {
      const item = document.createElement("button")
      item.type = "button"
      item.className = "private-background-option"
      if (background.path === selectedPath) item.classList.add("active")

      const image = document.createElement("img")
      image.src = background.src
      image.alt = ""
      image.loading = "lazy"
      image.setAttribute("aria-hidden", "true")

      const label = document.createElement("span")
      label.textContent = background.name

      item.appendChild(image)
      item.appendChild(label)
      item.addEventListener("click", () => selectBackground(background))
      grid.appendChild(item)
    }
  }

  const bindSettings = () => {
    const root = getRoot()
    if (!(root instanceof HTMLElement) || root.dataset.privateBackgroundBound === "true") return
    root.dataset.privateBackgroundBound = "true"

    getToggle()?.addEventListener("click", () => setEnabled(!isEnabled()))
    getSlideshowToggle()?.addEventListener("click", () => setSlideshowEnabled(!isSlideshowEnabled()))
    getSelectButton()?.addEventListener("click", openModal)

    getOpacitySlider()?.addEventListener("input", (event) => {
      const target = event.target
      if (target instanceof HTMLInputElement) {
        setOpacity(Number.parseInt(target.value, 10))
      }
    })

    getSlideshowSecondsInput()?.addEventListener("input", (event) => {
      const target = event.target
      if (target instanceof HTMLInputElement) {
        setSlideshowSeconds(Number.parseInt(target.value, 10))
      }
    })

    getCloseButton()?.addEventListener("click", closeModal)

    getModal()?.addEventListener("click", (event) => {
      if (event.target === getModal()) closeModal()
    })

    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") closeModal()
    })

    document.addEventListener(CHANGE_EVENT_NAME, () => {
      updateCurrentLabel()
      updateControls()
      renderGrid()
    })

    document.addEventListener(CHARACTER_CHANGE_EVENT_NAME, () => {
      updateCurrentLabel()
      updateControls()
      renderGrid()
    })
  }

  const init = () => {
    bindSettings()
    renderGrid()
    updateCurrentLabel()
    updateControls()
  }

  document.addEventListener("nav", init)
  init()
})()
`

PrivateBackgroundSettings.css = styles

export default (() => PrivateBackgroundSettings) satisfies QuartzComponentConstructor
