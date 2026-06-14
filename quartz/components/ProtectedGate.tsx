import styles from "./styles/protectedGate.scss"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import { pathToRoot, joinSegments } from "../util/path"
import {
  protectPasswordHash,
  protectPasswordSalt,
  protectPasswordLength,
  protectKdfIterations,
} from "../protect.generated"

const ProtectedGate: QuartzComponent = ({ fileData, displayClass }: QuartzComponentProps) => {
  const iconPath = joinSegments(pathToRoot(fileData.slug!), "static/action/protect.svg")

  return (
    <>
      <button
        type="button"
        id="protect-gate-trigger"
        className={`protect-gate-trigger ${displayClass ?? ""}`}
        aria-label="Protected feature gate"
      />
      <img
        id="protect-gate-indicator"
        className="protect-gate-indicator"
        src={iconPath}
        alt=""
        aria-hidden="true"
        hidden
      />
    </>
  )
}

const clientConfig = {
  hash: protectPasswordHash,
  salt: protectPasswordSalt,
  length: protectPasswordLength,
  iterations: protectKdfIterations,
}

ProtectedGate.afterDOMLoaded = `
(() => {
  const CONFIG = ${JSON.stringify(clientConfig)}
  const PRIVATE_STORAGE_KEY = "protect-authorized"
  const IMAGE_STORAGE_KEY = "show-images"
  const LEGACY_IMAGE_STORAGE_KEY = "protect"
  const PRIVATE_BODY_CLASS = "private-mode"
  const IMAGE_BODY_CLASS = "show-images"
  const PASSWORD_CONTEXT = "quartz-protect-v1"

  // Keep state inside this closure. The only global page state is body.protect after unlock.
  const state = window.__quartzProtectedGate ?? {
    armed: false,
    buffer: "",
    verifying: false,
    bound: false,
  }
  window.__quartzProtectedGate = state

  const getTrigger = () => document.getElementById("protect-gate-trigger")
  const getIndicator = () => document.getElementById("protect-gate-indicator")

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

  const setIndicatorVisible = (visible) => {
    const indicator = getIndicator()
    if (indicator) indicator.hidden = !visible
  }

  const migrateLegacyImageState = () => {
    if (safeGetStorage(IMAGE_STORAGE_KEY) !== null) return

    const legacyValue = safeGetStorage(LEGACY_IMAGE_STORAGE_KEY)
    if (legacyValue === "true" || legacyValue === "false") {
      safeSetStorage(IMAGE_STORAGE_KEY, legacyValue)
    }
  }

  const isPrivateMode = () => {
    return safeGetStorage(PRIVATE_STORAGE_KEY) === "true"
  }

  const isImageVisible = () => {
    return safeGetStorage(IMAGE_STORAGE_KEY) === "true"
  }

  const setPrivateMode = (enabled) => {
    if (enabled) {
      document.body.classList.add(PRIVATE_BODY_CLASS)
      safeSetStorage(PRIVATE_STORAGE_KEY, "true")
    } else {
      document.body.classList.remove(PRIVATE_BODY_CLASS)
      safeSetStorage(PRIVATE_STORAGE_KEY, "false")
    }
  }

  const setImageVisible = (visible) => {
    if (visible) {
      document.body.classList.add(IMAGE_BODY_CLASS)
      safeSetStorage(IMAGE_STORAGE_KEY, "true")
    } else {
      document.body.classList.remove(IMAGE_BODY_CLASS)
      safeSetStorage(IMAGE_STORAGE_KEY, "false")
    }
  }

  const applyStoredState = () => {
    migrateLegacyImageState()

    if (isPrivateMode()) {
      document.body.classList.add(PRIVATE_BODY_CLASS)
    } else {
      document.body.classList.remove(PRIVATE_BODY_CLASS)
    }

    if (isPrivateMode() && isImageVisible()) {
      document.body.classList.add(IMAGE_BODY_CLASS)
    } else {
      document.body.classList.remove(IMAGE_BODY_CLASS)
    }

    document.body.classList.remove(LEGACY_IMAGE_STORAGE_KEY)

    state.armed = false
    state.buffer = ""
    state.verifying = false
    setIndicatorVisible(false)
  }

  const toggleImages = () => {
    if (!isPrivateMode()) return
    setImageVisible(!document.body.classList.contains(IMAGE_BODY_CLASS))
    disarm()
  }

  const arm = () => {
    if (isPrivateMode()) {
      toggleImages()
      return
    }

    if (!CONFIG.hash || !CONFIG.salt || CONFIG.length <= 0) return
    state.armed = true
    state.buffer = ""
    state.verifying = false
    setIndicatorVisible(true)
  }

  const disarm = () => {
    state.armed = false
    state.buffer = ""
    state.verifying = false
    setIndicatorVisible(false)
  }

  const unlock = () => {
    setPrivateMode(true)
    setImageVisible(true)
    disarm()
  }

  const hexToBytes = (hex) => {
    const bytes = new Uint8Array(hex.length / 2)
    for (let i = 0; i < bytes.length; i++) {
      bytes[i] = Number.parseInt(hex.slice(i * 2, i * 2 + 2), 16)
    }
    return bytes
  }

  const bytesToHex = (bytes) => {
    return Array.from(bytes)
      .map((byte) => byte.toString(16).padStart(2, "0"))
      .join("")
  }

  const deriveHash = async (password) => {
    if (!window.crypto?.subtle) return null
    const encoder = new TextEncoder()
    const material = await crypto.subtle.importKey(
      "raw",
      encoder.encode(PASSWORD_CONTEXT + ":" + password),
      "PBKDF2",
      false,
      ["deriveBits"],
    )
    const bits = await crypto.subtle.deriveBits(
      {
        name: "PBKDF2",
        salt: hexToBytes(CONFIG.salt),
        iterations: CONFIG.iterations,
        hash: "SHA-256",
      },
      material,
      256,
    )
    return bytesToHex(new Uint8Array(bits))
  }

  const verifyBuffer = async () => {
    const attempt = state.buffer
    state.verifying = true
    try {
      const hash = await deriveHash(attempt)
      if (state.armed && state.buffer === attempt && hash === CONFIG.hash) {
        unlock()
      } else {
        disarm()
      }
    } catch {
      disarm()
    }
  }

  const isTriggerEvent = (event) => {
    const target = event.target
    return target instanceof Element && target.closest("#protect-gate-trigger")
  }

  const bindListenersOnce = () => {
    if (state.bound) return
    state.bound = true

    document.addEventListener("nav", applyStoredState)

    document.addEventListener("click", (event) => {
      if (isTriggerEvent(event)) arm()
    })

    document.addEventListener(
      "pointerdown",
      (event) => {
        if (!state.armed || isTriggerEvent(event)) return
        disarm()
      },
      true,
    )

    for (const eventName of ["wheel", "scroll", "touchstart", "paste", "compositionstart"]) {
      document.addEventListener(eventName, () => {
        if (state.armed) disarm()
      }, true)
    }

    document.addEventListener("keydown", (event) => {
      if (!state.armed || state.verifying) return

      if (event.key === "Escape") {
        disarm()
        return
      }

      // Any shortcut, control key, or non-character key is treated as accidental activation.
      if (event.ctrlKey || event.metaKey || event.altKey || event.key.length !== 1) {
        disarm()
        return
      }

      state.buffer += event.key

      if (state.buffer.length > CONFIG.length) {
        disarm()
        return
      }

      if (state.buffer.length === CONFIG.length) {
        verifyBuffer()
      }
    })
  }

  bindListenersOnce()
  applyStoredState()
})()
`

ProtectedGate.css = styles

export default (() => ProtectedGate) satisfies QuartzComponentConstructor
