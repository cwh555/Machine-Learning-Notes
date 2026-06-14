import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import { pathToRoot, joinSegments } from "../util/path"
import { getPrivateAnimationCharacters } from "../util/privateAnimations"

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

const PrivateAnimationSettings: QuartzComponent = ({ fileData }: QuartzComponentProps) => {
  const characters = getPrivateAnimationCharacters().map((character) => ({
    name: character.name,
    frames: character.frames.map((frame) => ({
      name: frame.name,
      src: joinSegments(pathToRoot(fileData.slug!), "static/animation", character.name, frame.name),
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

      <div id="private-animation-details" class="private-animation-details" hidden>
        <div class="private-setting-row private-animation-character-row">
          <span class="private-setting-label">character:</span>
          <div id="private-animation-character-list" class="private-animation-character-list" />
        </div>

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
  const DEFAULT_SIZE = ${JSON.stringify(defaultSize)}
  const DEFAULT_OPACITY = ${JSON.stringify(defaultOpacity)}
  const DEFAULT_FPS = ${JSON.stringify(defaultFps)}
  const DEFAULT_MIN_GUARD = ${JSON.stringify(defaultMinGuard)}

  const getRoot = () => document.querySelector(".private-animation-settings")
  const getToggle = () => document.getElementById("private-animation-toggle")
  const getToggleLabel = () => document.getElementById("private-animation-toggle-label")
  const getDetails = () => document.getElementById("private-animation-details")
  const getCharacterList = () => document.getElementById("private-animation-character-list")
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
  const getSelectedCharacterName = () => safeGetStorage(CHARACTER_STORAGE_KEY)
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

    if (toggle instanceof HTMLButtonElement) {
      toggle.setAttribute("aria-checked", enabled ? "true" : "false")
      toggle.classList.toggle("active", enabled)
    }

    if (label) label.textContent = enabled ? "on" : "off"
    if (details) details.hidden = !enabled
  }

  const updateSlider = (slider, label, value, suffix = "") => {
    if (slider instanceof HTMLInputElement) {
      slider.value = String(value)
    }

    if (label) {
      label.textContent = String(value) + suffix
    }
  }

  const updateControls = () => {
    updateSwitch()
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

  const selectCharacter = (name) => {
    safeSetStorage(CHARACTER_STORAGE_KEY, name)
    renderCharacters()
    emitChange()
  }

  const renderEmptyState = (list) => {
    const empty = document.createElement("p")
    empty.className = "private-animation-empty"
    empty.textContent = "No animation characters found."
    list.appendChild(empty)
  }

  const renderCharacters = () => {
    const list = getCharacterList()
    if (!list) return

    list.textContent = ""
    const characters = parseCharacters()
    if (characters.length === 0) {
      renderEmptyState(list)
      return
    }

    const selectedName = getSelectedCharacterName()
    for (const character of characters) {
      const row = document.createElement("button")
      row.type = "button"
      row.className = "private-animation-character-option"
      if (character.name === selectedName) row.classList.add("active")

      const name = document.createElement("span")
      name.className = "private-animation-character-name"
      name.textContent = character.name
      row.appendChild(name)

      const frames = document.createElement("span")
      frames.className = "private-animation-frame-strip"

      for (const frame of character.frames ?? []) {
        const image = document.createElement("img")
        image.src = frame.src
        image.alt = ""
        image.loading = "lazy"
        image.setAttribute("aria-hidden", "true")
        frames.appendChild(image)
      }

      row.appendChild(frames)
      row.addEventListener("click", () => selectCharacter(character.name))
      list.appendChild(row)
    }
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
  }

  const init = () => {
    bindSettings()
    renderCharacters()
    updateControls()
  }

  document.addEventListener("nav", init)
  init()
})()
`

PrivateAnimationSettings.css = `
.private-animation-settings {
  display: none;
  margin-top: 1rem;
  padding: 1rem;
  border: 1px solid var(--lightgray);
  border-radius: 8px;
  background: var(--light);
}

body.private-mode .private-animation-settings {
  display: block;
}

.private-animation-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  margin-bottom: 0.75rem;
}

.private-animation-switch {
  display: inline-flex;
  align-items: center;
  gap: 0.45rem;
  border: 0;
  background: none;
  color: var(--darkgray);
  cursor: pointer;
  font: inherit;
  padding: 0;
}

.private-animation-switch-track {
  position: relative;
  width: 2.4rem;
  height: 1.2rem;
  border: 1px solid var(--lightgray);
  border-radius: 999px;
  background: var(--lightgray);
  transition: background 0.15s ease, border-color 0.15s ease;
}

.private-animation-switch-knob {
  position: absolute;
  top: 50%;
  left: 0.15rem;
  width: 0.9rem;
  height: 0.9rem;
  border-radius: 999px;
  background: var(--light);
  transform: translateY(-50%);
  transition: left 0.15s ease;
}

.private-animation-switch.active .private-animation-switch-track {
  border-color: var(--secondary);
  background: var(--secondary);
}

.private-animation-switch.active .private-animation-switch-knob {
  left: 1.25rem;
}

.private-animation-switch-label {
  min-width: 1.8rem;
  color: var(--darkgray);
}

.private-animation-details {
  display: grid;
  gap: 0.65rem;
}

.private-animation-details[hidden] {
  display: none !important;
}

.private-animation-character-row {
  align-items: flex-start;
}

.private-animation-character-list {
  display: grid;
  gap: 0.5rem;
  flex: 1 1 100%;
}

.private-animation-character-option {
  display: grid;
  grid-template-columns: minmax(5rem, max-content) 1fr;
  align-items: center;
  gap: 0.75rem;
  width: 100%;
  padding: 0.5rem;
  border: 1px solid var(--lightgray);
  border-radius: 8px;
  background: var(--light);
  color: var(--darkgray);
  cursor: pointer;
  font: inherit;
  text-align: left;
}

.private-animation-character-option:hover,
.private-animation-character-option.active {
  border-color: var(--secondary);
  color: var(--secondary);
}

.private-animation-character-name {
  font-weight: 600;
}

.private-animation-frame-strip {
  display: flex;
  align-items: center;
  gap: 0.35rem;
  overflow-x: auto;
}

.private-animation-frame-strip img {
  width: 34px;
  height: 34px;
  object-fit: contain;
  flex: 0 0 auto;
}

.private-animation-control-row {
  align-items: baseline;
}

.private-animation-slider {
  width: min(240px, 100%);
  accent-color: var(--secondary);
}

.private-animation-empty {
  margin: 0;
  color: var(--gray);
}
`

export default (() => PrivateAnimationSettings) satisfies QuartzComponentConstructor
