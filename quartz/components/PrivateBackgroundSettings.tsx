import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import { pathToRoot, joinSegments } from "../util/path"
import { getPrivateBackgroundFiles } from "../util/privateBackgrounds"

const storageKey = "private-background"
const opacityStorageKey = "private-background-opacity"
const defaultOpacity = 15
const changeEventName = "private-background-change"

const PrivateBackgroundSettings: QuartzComponent = ({ fileData }: QuartzComponentProps) => {
  const backgrounds = getPrivateBackgroundFiles().map((background) => ({
    name: background.name,
    src: joinSegments(pathToRoot(fileData.slug!), "static/background", background.name),
  }))

  return (
    <section class="private-background-settings" data-backgrounds={JSON.stringify(backgrounds)}>
      <h2 class="private-setting-subtitle">Background</h2>

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
  const STORAGE_KEY = ${JSON.stringify(storageKey)}
  const OPACITY_STORAGE_KEY = ${JSON.stringify(opacityStorageKey)}
  const DEFAULT_OPACITY = ${JSON.stringify(defaultOpacity)}
  const CHANGE_EVENT_NAME = ${JSON.stringify(changeEventName)}

  const getRoot = () => document.querySelector(".private-background-settings")
  const getCurrent = () => document.getElementById("private-background-current")
  const getSelectButton = () => document.getElementById("private-background-select")
  const getOpacitySlider = () => document.getElementById("private-background-opacity")
  const getOpacityValue = () => document.getElementById("private-background-opacity-value")
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

  const clampOpacity = (value) => {
    if (!Number.isFinite(value)) return DEFAULT_OPACITY
    return Math.min(100, Math.max(0, Math.round(value)))
  }

  const getOpacity = () => {
    const raw = safeGetStorage(OPACITY_STORAGE_KEY)
    return clampOpacity(Number.parseInt(raw ?? "", 10))
  }

  const updateOpacityControl = () => {
    const opacity = getOpacity()
    const slider = getOpacitySlider()
    const label = getOpacityValue()

    if (slider instanceof HTMLInputElement) {
      slider.value = String(opacity)
    }

    if (label) {
      label.textContent = opacity + "%"
    }
  }

  const setOpacity = (value) => {
    const opacity = clampOpacity(value)
    safeSetStorage(OPACITY_STORAGE_KEY, String(opacity))
    updateOpacityControl()
    document.dispatchEvent(new CustomEvent(CHANGE_EVENT_NAME, { detail: { opacity } }))
  }


  const parseBackgrounds = () => {
    const root = getRoot()
    if (!(root instanceof HTMLElement)) return []

    try {
      const raw = root.dataset.backgrounds ?? "[]"
      const parsed = JSON.parse(raw)
      return Array.isArray(parsed) ? parsed : []
    } catch {
      return []
    }
  }

  const getSelectedName = () => safeGetStorage(STORAGE_KEY)

  const updateCurrentLabel = () => {
    const current = getCurrent()
    if (!current) return

    const selectedName = getSelectedName()
    const selected = parseBackgrounds().find((background) => background.name === selectedName)
    current.textContent = selected?.name ?? "none"
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

  const selectBackground = (name) => {
    safeSetStorage(STORAGE_KEY, name)
    updateCurrentLabel()
    document.dispatchEvent(new CustomEvent(CHANGE_EVENT_NAME, { detail: { name } }))
    closeModal()
  }

  const renderEmptyState = (grid) => {
    const empty = document.createElement("p")
    empty.className = "private-background-empty"
    empty.textContent = "No background images found."
    grid.appendChild(empty)
  }

  const renderGrid = () => {
    const grid = getGrid()
    if (!grid) return

    grid.textContent = ""
    const backgrounds = parseBackgrounds()
    if (backgrounds.length === 0) {
      renderEmptyState(grid)
      return
    }

    const selectedName = getSelectedName()
    for (const background of backgrounds) {
      const item = document.createElement("button")
      item.type = "button"
      item.className = "private-background-option"
      if (background.name === selectedName) item.classList.add("active")

      const image = document.createElement("img")
      image.src = background.src
      image.alt = ""
      image.loading = "lazy"
      image.setAttribute("aria-hidden", "true")

      const label = document.createElement("span")
      label.textContent = background.name

      item.appendChild(image)
      item.appendChild(label)
      item.addEventListener("click", () => selectBackground(background.name))
      grid.appendChild(item)
    }
  }

  const bindSettings = () => {
    const root = getRoot()
    if (!(root instanceof HTMLElement) || root.dataset.privateBackgroundBound === "true") return
    root.dataset.privateBackgroundBound = "true"

    getSelectButton()?.addEventListener("click", openModal)

    getOpacitySlider()?.addEventListener("input", (event) => {
      const target = event.target
      if (target instanceof HTMLInputElement) {
        setOpacity(Number.parseInt(target.value, 10))
      }
    })

    getCloseButton()?.addEventListener("click", closeModal)

    getModal()?.addEventListener("click", (event) => {
      if (event.target === getModal()) closeModal()
    })

    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") closeModal()
    })
  }

  const init = () => {
    bindSettings()
    updateCurrentLabel()
    updateOpacityControl()
  }

  document.addEventListener("nav", init)
  init()
})()
`

PrivateBackgroundSettings.css = `
.private-background-settings {
  display: none;
  margin-top: 1rem;
  padding: 1rem;
  border: 1px solid var(--lightgray);
  border-radius: 8px;
  background: var(--light);
}

body.private-mode .private-background-settings {
  display: block;
}

.private-setting-subtitle {
  margin: 0 0 0.75rem;
  font-size: 1.15rem;
  line-height: 1.3;
  color: var(--dark);
}

.private-setting-row {
  display: flex;
  align-items: baseline;
  gap: 0.5rem;
  flex-wrap: wrap;
  font: inherit;
  line-height: 1.4;
}

.private-setting-label {
  color: var(--darkgray);
  font-weight: 600;
}

.private-setting-current {
  color: var(--secondary);
  font: inherit;
}

.private-setting-button,
.private-background-close,
.private-background-option {
  cursor: pointer;
  border: 1px solid var(--lightgray);
  border-radius: 6px;
  background: var(--light);
  color: var(--darkgray);
  font: inherit;
}

.private-setting-button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 0.1rem 0.45rem;
  line-height: 1.4;
}

.private-opacity-row {
  margin-top: 0.6rem;
}

.private-opacity-slider {
  width: min(220px, 100%);
  accent-color: var(--secondary);
}

.private-setting-button:hover,
.private-background-close:hover,
.private-background-option:hover,
.private-background-option.active {
  border-color: var(--secondary);
  color: var(--secondary);
}

.private-background-modal {
  position: fixed;
  inset: 0;
  z-index: 100000;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 1rem;
  background: rgba(0, 0, 0, 0.18);
}

.private-background-modal[hidden] {
  display: none !important;
}

.private-background-panel {
  width: min(720px, calc(100vw - 2rem));
  max-height: min(640px, calc(100vh - 2rem));
  overflow: auto;
  border: 1px solid var(--lightgray);
  border-radius: 8px;
  background: var(--light);
  color: var(--darkgray);
  box-shadow: 0 14px 40px rgba(0, 0, 0, 0.18);
  padding: 1rem;
}

.private-background-panel-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  margin-bottom: 1rem;
}

.private-background-panel-header h2 {
  margin: 0;
  font-size: 1.2rem;
}

.private-background-close {
  width: 2rem;
  height: 2rem;
  font-size: 1.2rem;
  line-height: 1;
}

.private-background-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
  gap: 0.8rem;
}

.private-background-option {
  display: flex;
  flex-direction: column;
  gap: 0.45rem;
  padding: 0.45rem;
  text-align: left;
}

.private-background-option img {
  width: 100%;
  aspect-ratio: 16 / 10;
  object-fit: cover;
  border-radius: 4px;
}

.private-background-option span {
  overflow-wrap: anywhere;
  font-size: 0.85rem;
}

.private-background-empty {
  margin: 0;
  color: var(--gray);
}
`

export default (() => PrivateBackgroundSettings) satisfies QuartzComponentConstructor
