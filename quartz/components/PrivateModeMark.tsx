import styles from "./styles/privateModeMark.scss"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import { pathToRoot, joinSegments } from "../util/path"
import { getPrivateCharacters } from "../util/privateCharacters"

const characterStorageKey = "private-character"
const oldAnimationCharacterStorageKey = "private-animation-character"
const backgroundEnabledStorageKey = "private-background-enabled"
const backgroundStorageKey = "private-background"
const backgroundOpacityStorageKey = "private-background-opacity"
const slideshowEnabledStorageKey = "private-background-slideshow-enabled"
const slideshowSecondsStorageKey = "private-background-slideshow-seconds"
const animationEnabledStorageKey = "private-animation-enabled"
const animationSizeStorageKey = "private-animation-size"
const animationOpacityStorageKey = "private-animation-opacity"
const animationFpsStorageKey = "private-animation-fps"
const animationMinGuardStorageKey = "private-animation-min-guard"
const privateModeClass = "private-mode"
const characterChangeEventName = "private-character-change"
const backgroundChangeEventName = "private-background-change"
const animationChangeEventName = "private-animation-change"

const PrivateModeMark: QuartzComponent = ({ fileData }: QuartzComponentProps) => {
  const characters = getPrivateCharacters().map((character) => ({
    id: character.id,
    name: character.name,
    defaults: character.defaults,
    backgrounds: character.backgrounds.map((background) => ({
      name: background.name,
      path: background.path,
    })),
    frames: character.animation.frames.map((frame) => ({
      name: frame.name,
      path: frame.path,
    })),
    mark: character.mark
      ? {
          name: character.mark.name,
          path: character.mark.path,
          src: joinSegments(pathToRoot(fileData.slug!), "static/characters", character.id, character.mark.path),
        }
      : null,
  }))

  return (
    <div id="private-mode-mark-root" data-characters={JSON.stringify(characters)} hidden>
      <button
        id="private-mode-mark"
        type="button"
        aria-label="Switch private character"
        aria-haspopup="menu"
        aria-expanded="false"
      >
        <img id="private-mode-mark-image" alt="" aria-hidden="true" />
      </button>
      <div id="private-mode-character-menu" role="menu" hidden />
    </div>
  )
}

PrivateModeMark.afterDOMLoaded = `
(() => {
  const CHARACTER_STORAGE_KEY = ${JSON.stringify(characterStorageKey)}
  const OLD_ANIMATION_CHARACTER_STORAGE_KEY = ${JSON.stringify(oldAnimationCharacterStorageKey)}
  const BACKGROUND_ENABLED_STORAGE_KEY = ${JSON.stringify(backgroundEnabledStorageKey)}
  const BACKGROUND_STORAGE_KEY = ${JSON.stringify(backgroundStorageKey)}
  const BACKGROUND_OPACITY_STORAGE_KEY = ${JSON.stringify(backgroundOpacityStorageKey)}
  const SLIDESHOW_ENABLED_STORAGE_KEY = ${JSON.stringify(slideshowEnabledStorageKey)}
  const SLIDESHOW_SECONDS_STORAGE_KEY = ${JSON.stringify(slideshowSecondsStorageKey)}
  const ANIMATION_ENABLED_STORAGE_KEY = ${JSON.stringify(animationEnabledStorageKey)}
  const ANIMATION_SIZE_STORAGE_KEY = ${JSON.stringify(animationSizeStorageKey)}
  const ANIMATION_OPACITY_STORAGE_KEY = ${JSON.stringify(animationOpacityStorageKey)}
  const ANIMATION_FPS_STORAGE_KEY = ${JSON.stringify(animationFpsStorageKey)}
  const ANIMATION_MIN_GUARD_STORAGE_KEY = ${JSON.stringify(animationMinGuardStorageKey)}
  const PRIVATE_MODE_CLASS = ${JSON.stringify(privateModeClass)}
  const CHARACTER_CHANGE_EVENT_NAME = ${JSON.stringify(characterChangeEventName)}
  const BACKGROUND_CHANGE_EVENT_NAME = ${JSON.stringify(backgroundChangeEventName)}
  const ANIMATION_CHANGE_EVENT_NAME = ${JSON.stringify(animationChangeEventName)}

  const state = window.__quartzPrivateModeMark ?? {
    bound: false,
    observer: null,
  }
  window.__quartzPrivateModeMark = state

  const getRoot = () => document.getElementById("private-mode-mark-root")
  const getButton = () => document.getElementById("private-mode-mark")
  const getImage = () => document.getElementById("private-mode-mark-image")
  const getMenu = () => document.getElementById("private-mode-character-menu")

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

  const safeRemoveStorage = (key) => {
    try {
      localStorage.removeItem(key)
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

  const getMarkCharacters = () => parseCharacters().filter((character) => Boolean(character.mark?.src))

  const getSelectedCharacter = () => {
    const id = safeGetStorage(CHARACTER_STORAGE_KEY)
    if (!id) return null
    return parseCharacters().find((character) => character.id === id) ?? null
  }

  const closeMenu = () => {
    const button = getButton()
    const menu = getMenu()
    if (button instanceof HTMLButtonElement) button.setAttribute("aria-expanded", "false")
    if (menu instanceof HTMLElement) menu.hidden = true
  }

  const toggleMenu = () => {
    const button = getButton()
    const menu = getMenu()
    if (!(button instanceof HTMLButtonElement) || !(menu instanceof HTMLElement)) return

    const nextOpen = menu.hidden
    menu.hidden = !nextOpen
    button.setAttribute("aria-expanded", String(nextOpen))
  }

  const emitProfileChange = (character) => {
    document.dispatchEvent(new CustomEvent(CHARACTER_CHANGE_EVENT_NAME, { detail: { id: character.id, name: character.name } }))
    document.dispatchEvent(new CustomEvent(BACKGROUND_CHANGE_EVENT_NAME, { detail: { character: character.id, source: "mark" } }))
    document.dispatchEvent(new CustomEvent(ANIMATION_CHANGE_EVENT_NAME, { detail: { character: character.id, source: "mark" } }))
  }

  const applyCharacter = (character) => {
    const defaults = character.defaults ?? {}
    const firstBackground = character.backgrounds?.[0]
    const hasBackgrounds = Array.isArray(character.backgrounds) && character.backgrounds.length > 0
    const hasFrames = Array.isArray(character.frames) && character.frames.length > 0

    safeSetStorage(CHARACTER_STORAGE_KEY, character.id)
    safeRemoveStorage(OLD_ANIMATION_CHARACTER_STORAGE_KEY)

    safeSetStorage(BACKGROUND_ENABLED_STORAGE_KEY, defaults.backgroundEnabled && hasBackgrounds ? "true" : "false")
    if (firstBackground?.path) {
      safeSetStorage(BACKGROUND_STORAGE_KEY, firstBackground.path)
    } else {
      safeRemoveStorage(BACKGROUND_STORAGE_KEY)
    }
    safeSetStorage(BACKGROUND_OPACITY_STORAGE_KEY, String(defaults.backgroundOpacity ?? 15))
    safeSetStorage(SLIDESHOW_ENABLED_STORAGE_KEY, defaults.slideshowEnabled && hasBackgrounds ? "true" : "false")
    safeSetStorage(SLIDESHOW_SECONDS_STORAGE_KEY, String(defaults.slideshowSeconds ?? 8))

    safeSetStorage(ANIMATION_ENABLED_STORAGE_KEY, defaults.animationEnabled && hasFrames ? "true" : "false")
    safeSetStorage(ANIMATION_SIZE_STORAGE_KEY, String(defaults.animationSize ?? 100))
    safeSetStorage(ANIMATION_OPACITY_STORAGE_KEY, String(defaults.animationOpacity ?? 100))
    safeSetStorage(ANIMATION_FPS_STORAGE_KEY, String(defaults.animationFps ?? 8))
    safeSetStorage(ANIMATION_MIN_GUARD_STORAGE_KEY, String(defaults.animationMinGuard ?? 1200))

    updateMark()
    renderMenu()
    closeMenu()
    emitProfileChange(character)
  }

  const renderMenu = () => {
    const menu = getMenu()
    if (!(menu instanceof HTMLElement)) return

    const selected = safeGetStorage(CHARACTER_STORAGE_KEY)
    const characters = getMarkCharacters()
    menu.textContent = ""

    for (const character of characters) {
      const item = document.createElement("button")
      item.type = "button"
      item.className = "private-mode-character-menu-item"
      item.setAttribute("role", "menuitem")
      item.setAttribute("aria-label", "Switch to " + character.name)
      item.title = character.name
      if (character.id === selected) item.classList.add("active")

      const image = document.createElement("img")
      image.src = character.mark.src
      image.alt = ""
      image.loading = "lazy"
      image.setAttribute("aria-hidden", "true")
      item.appendChild(image)

      item.addEventListener("click", (event) => {
        event.preventDefault()
        event.stopPropagation()
        applyCharacter(character)
      })

      menu.appendChild(item)
    }
  }

  const updateMark = () => {
    const root = getRoot()
    const button = getButton()
    const image = getImage()
    if (!(root instanceof HTMLElement) || !(button instanceof HTMLButtonElement) || !(image instanceof HTMLImageElement)) return

    const character = getSelectedCharacter()
    const markSrc = character?.mark?.src
    const visible = document.body.classList.contains(PRIVATE_MODE_CLASS) && Boolean(markSrc)

    root.hidden = !visible
    if (!visible) {
      image.removeAttribute("src")
      button.removeAttribute("title")
      closeMenu()
      return
    }

    if (image.src !== new URL(markSrc, window.location.href).href) {
      image.src = markSrc
    }
    button.title = character?.name ? "Switch private character: " + character.name : "Switch private character"
  }

  const bindListenersOnce = () => {
    if (state.bound) return
    state.bound = true

    document.addEventListener("nav", () => {
      renderMenu()
      updateMark()
    })
    document.addEventListener(CHARACTER_CHANGE_EVENT_NAME, () => {
      renderMenu()
      updateMark()
    })

    document.addEventListener("click", (event) => {
      const root = getRoot()
      if (root instanceof HTMLElement && !root.contains(event.target)) closeMenu()
    })

    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") closeMenu()
    })

    const button = getButton()
    if (button instanceof HTMLButtonElement) {
      button.addEventListener("click", (event) => {
        event.preventDefault()
        event.stopPropagation()
        toggleMenu()
      })
    }

    state.observer = new MutationObserver(() => {
      renderMenu()
      updateMark()
    })
    state.observer.observe(document.body, { attributes: true, attributeFilter: ["class"] })
  }

  bindListenersOnce()
  renderMenu()
  updateMark()
})()
`

PrivateModeMark.css = styles

export default (() => PrivateModeMark) satisfies QuartzComponentConstructor