import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"

const ImageToggle: QuartzComponent = ({ displayClass }: QuartzComponentProps) => {
  return (
    <button
      type="button"
      id="image-toggle"
      className={`image-toggle ${displayClass ?? ""}`}
      aria-label="Toggle Image"
    >
      <svg
        xmlns="http://www.w3.org/2000/svg"
        width="20"
        height="20"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        className="image-icon"
      >
        <rect width="18" height="18" x="3" y="3" rx="2" ry="2" />
        <circle cx="9" cy="9" r="2" />
        <path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21" />
      </svg>
    </button>
  )
}

// 監聽 nav 事件，確保換頁後功能依然正常
ImageToggle.afterDOMLoaded = `
document.addEventListener("nav", () => {
  const toggleBtn = document.getElementById("image-toggle")
  const body = document.body
  
  if (!toggleBtn) return

  // 1. 初始化狀態
  const storedState = localStorage.getItem("show-images")
  const showImages = storedState === "true"

  if (showImages) {
    body.classList.add("show-images")
    toggleBtn.classList.add("active")
  } else {
    body.classList.remove("show-images")
    toggleBtn.classList.remove("active")
  }

  // 2. 點擊處理
  const handleClick = () => {
    const isShowing = body.classList.toggle("show-images")
    toggleBtn.classList.toggle("active")
    localStorage.setItem("show-images", isShowing.toString())
  }

  toggleBtn.removeEventListener("click", handleClick) 
  toggleBtn.addEventListener("click", handleClick)
})
`

// --- 修改這裡：改成隱形樣式 ---
ImageToggle.css = `
.image-toggle {
  background: none;
  border: none;
  padding: 0;
  margin-left: 0.5rem;
  display: flex;
  align-items: center;

  /* 關鍵設定：完全隱形 */
  opacity: 0;
  
  /* 關鍵設定：滑鼠滑過去是普通的箭頭，不是手指，偽裝成空白處 */
  cursor: default; 
  
  /* 確保按鈕還有體積，不然會點不到 (20x20px) */
  width: 20px;
  height: 20px;
}

/* 滑鼠滑過去依然保持隱形 */
.image-toggle:hover {
  opacity: 0;
}

/* 就算被啟動了，也依然保持隱形 */
.image-toggle.active {
  opacity: 0;
}
`

export default (() => ImageToggle) satisfies QuartzComponentConstructor