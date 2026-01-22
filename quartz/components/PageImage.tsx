import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"

const PageImage: QuartzComponent = ({ fileData, displayClass }: QuartzComponentProps) => {
  // 1. 讀取 frontmatter 中的 image 欄位
  const imageName = fileData.frontmatter?.image

  // 2. 如果沒有設定 image，就不顯示
  if (!imageName) {
    return null
  }

  // 3. 組合圖片路徑 (指向 static/images/)
  const imagePath = `/static/images/${imageName}`

  return (
    // 修正點：不使用 classNames 工具，直接用字串拼接，避免報錯
    <div className={`page-image ${displayClass ?? ""}`}>
      <img 
        src={imagePath} 
        alt={fileData.frontmatter?.title || "Featured Image"} 
        style={{
          width: "100%",
          borderRadius: "8px",
          marginBottom: "1rem",
          objectFit: "cover"
        }}
      />
    </div>
  )
}

export default (() => PageImage) satisfies QuartzComponentConstructor