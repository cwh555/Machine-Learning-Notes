import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import { pathToRoot, joinSegments } from "../util/path"

const PageImage: QuartzComponent = ({ fileData, displayClass }: QuartzComponentProps) => {
  const imageName = fileData.frontmatter?.image
  if (!imageName) return null

  // 1. 防呆處理：如果你在 frontmatter 寫了 "/0001.jpg" (有斜線)，這裡幫你去掉
  // 避免 joinSegments 拼接出錯誤路徑
  const cleanImageName = imageName.startsWith('/') ? imageName.slice(1) : imageName

  // 2. 使用 Quartz 原生方式計算「相對路徑」
  // 這是唯一的通用解，不用寫死 repo 名稱
  // Localhost (根目錄): 變成 "./static/images/xxx.jpg"
  // GitHub Pages (子目錄): 變成 "../../static/images/xxx.jpg" (自動倒退回根目錄)
  const imagePath = joinSegments(pathToRoot(fileData.slug!), "static/images", cleanImageName)

  return (
    <div className={`page-image ${displayClass ?? ""}`}>
      <img 
        src={imagePath} 
        alt={fileData.frontmatter?.title || "Featured Image"} 
        style={{
          width: "100%",
          borderRadius: "8px",
          marginBottom: "1rem",
          height: "auto",
          display: "block"
        }}
      />
    </div>
  )
}

export default (() => PageImage) satisfies QuartzComponentConstructor