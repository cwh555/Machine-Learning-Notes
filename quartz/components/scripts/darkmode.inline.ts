const userPref = window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark"
const currentTheme = localStorage.getItem("theme") ?? userPref
document.documentElement.setAttribute("saved-theme", currentTheme)

const PRIVATE_MODE_CLASS = "private-mode"
const PROTECT_FLASH_CLASS = "protect-light-denied-flash"

const emitThemeChangeEvent = (theme: "light" | "dark") => {
  const event: CustomEventMap["themechange"] = new CustomEvent("themechange", {
    detail: { theme },
  })
  document.dispatchEvent(event)
}

const setTheme = (theme: "light" | "dark") => {
  document.documentElement.setAttribute("saved-theme", theme)
  localStorage.setItem("theme", theme)
  emitThemeChangeEvent(theme)
}

const isPrivateModeActive = () => document.body.classList.contains(PRIVATE_MODE_CLASS)

const getDeniedImageSrc = () => {
  const darkmodeButton = document.querySelector(".darkmode")
  if (!(darkmodeButton instanceof HTMLElement)) return null
  return darkmodeButton.dataset.protectDeniedSrc ?? null
}

const flashLightDeniedImage = () => {
  const src = getDeniedImageSrc()
  if (!src) return

  document.querySelectorAll(`.${PROTECT_FLASH_CLASS}`).forEach((node) => node.remove())

  const image = document.createElement("img")
  image.className = PROTECT_FLASH_CLASS
  image.src = src
  image.alt = ""
  image.setAttribute("aria-hidden", "true")
  image.addEventListener("animationend", () => image.remove(), { once: true })
  document.body.appendChild(image)
}

const forceDarkIfProtected = () => {
  if (isPrivateModeActive() && document.documentElement.getAttribute("saved-theme") !== "dark") {
    setTheme("dark")
  }
}

const watchProtectClass = () => {
  const state = (window as Window & { __quartzProtectThemeObserverBound?: boolean })
  if (state.__quartzProtectThemeObserverBound) return
  state.__quartzProtectThemeObserverBound = true

  const observer = new MutationObserver(forceDarkIfProtected)
  observer.observe(document.body, { attributes: true, attributeFilter: ["class"] })
}

document.addEventListener("nav", () => {
  const switchTheme = () => {
    const currentTheme = document.documentElement.getAttribute("saved-theme")
    const newTheme = currentTheme === "dark" ? "light" : "dark"

    // In protected mode, briefly show light mode, flash the denied image, then return to dark.
    if (isPrivateModeActive() && newTheme === "light") {
      setTheme("light")
      flashLightDeniedImage()

      window.setTimeout(() => {
        if (isPrivateModeActive()) {
          setTheme("dark")
        }
      }, 400)

      return
    }

    setTheme(newTheme)
  }

  const themeChange = (e: MediaQueryListEvent) => {
    const newTheme = e.matches ? "dark" : "light"
    if (isPrivateModeActive() && newTheme === "light") {
      setTheme("dark")
      return
    }

    setTheme(newTheme)
  }

  watchProtectClass()
  forceDarkIfProtected()

  for (const darkmodeButton of document.getElementsByClassName("darkmode")) {
    darkmodeButton.addEventListener("click", switchTheme)
    window.addCleanup(() => darkmodeButton.removeEventListener("click", switchTheme))
  }

  // Listen for changes in prefers-color-scheme
  const colorSchemeMediaQuery = window.matchMedia("(prefers-color-scheme: dark)")
  colorSchemeMediaQuery.addEventListener("change", themeChange)
  window.addCleanup(() => colorSchemeMediaQuery.removeEventListener("change", themeChange))
})
