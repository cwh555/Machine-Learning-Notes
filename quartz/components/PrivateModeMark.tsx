import styles from "./styles/privateModeMark.scss"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import { pathToRoot, joinSegments } from "../util/path"
import { getPrivateCharacters } from "../util/privateCharacters"

const characterStorageKey = "private-character"
const privateModeClass = "private-mode"
const characterChangeEventName = "private-character-change"

const PrivateModeMark: QuartzComponent = ({ fileData }: QuartzComponentProps) => {
  const characters = getPrivateCharacters().map((character) => ({
    id: character.id,
    name: character.name,
    mark: character.mark
      ? {
          name: character.mark.name,
          path: character.mark.path,
          src: joinSegments(pathToRoot(fileData.slug!), "static/characters", character.id, character.mark.path),
        }
      : null,
  }))
  const settingsHref = joinSegments(pathToRoot(fileData.slug!), "action-settings")

  return (
    <a
      id="private-mode-mark"
      href={settingsHref}
      data-characters={JSON.stringify(characters)}
      aria-label="Open private settings"
      hidden
    >
      <img id="private-mode-mark-image" alt="" aria-hidden="true" />
    </a>
  )
}

PrivateModeMark.afterDOMLoaded = `
(() => {
  const CHARACTER_STORAGE_KEY = ${JSON.stringify(characterStorageKey)}
  const PRIVATE_MODE_CLASS = ${JSON.stringify(privateModeClass)}
  const CHARACTER_CHANGE_EVENT_NAME = ${JSON.stringify(characterChangeEventName)}

  const state = window.__quartzPrivateModeMark ?? {
    bound: false,
    observer: null,
  }
  window.__quartzPrivateModeMark = state

  const getMark = () => document.getElementById("private-mode-mark")
  const getImage = () => document.getElementById("private-mode-mark-image")

  const safeGetStorage = (key) => {
    try {
      return localStorage.getItem(key)
    } catch {
      return null
    }
  }

  const parseCharacters = () => {
    const mark = getMark()
    if (!(mark instanceof HTMLElement)) return []

    try {
      const raw = mark.dataset.characters ?? "[]"
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

  const updateMark = () => {
    const mark = getMark()
    const image = getImage()
    if (!(mark instanceof HTMLAnchorElement) || !(image instanceof HTMLImageElement)) return

    const character = getSelectedCharacter()
    const markSrc = character?.mark?.src
    const visible = document.body.classList.contains(PRIVATE_MODE_CLASS) && Boolean(markSrc)

    mark.hidden = !visible
    if (!visible) {
      image.removeAttribute("src")
      mark.removeAttribute("title")
      return
    }

    if (image.src !== new URL(markSrc, window.location.href).href) {
      image.src = markSrc
    }
    mark.title = character?.name ? character.name + " private settings" : "Private settings"
  }

  const bindListenersOnce = () => {
    if (state.bound) return
    state.bound = true

    document.addEventListener("nav", updateMark)
    document.addEventListener(CHARACTER_CHANGE_EVENT_NAME, updateMark)

    state.observer = new MutationObserver(updateMark)
    state.observer.observe(document.body, { attributes: true, attributeFilter: ["class"] })
  }

  bindListenersOnce()
  updateMark()
})()
`

PrivateModeMark.css = styles

export default (() => PrivateModeMark) satisfies QuartzComponentConstructor
