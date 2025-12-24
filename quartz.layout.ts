import { PageLayout, SharedLayout } from "./quartz/cfg"
import * as Component from "./quartz/components"

// components shared across all pages
export const sharedPageComponents: SharedLayout = {
  head: Component.Head(),
  header: [],
  afterBody: [],
  footer: Component.Footer({
    links: {
      github: { text: "GitHub", url: "https://github.com/cwh555/Machine-Learning-Notes" }
    },
    text: {
      contact: { text: "Contact : bergerchen719@gmail.com" }
    }
  }),
}

// components for pages that display a single page (e.g. a single note)
export const defaultContentPageLayout: PageLayout = {
  beforeBody: [
    Component.ConditionalRender({
      component: Component.Breadcrumbs(),
      // Hide breadcrumbs on the home page (index)
      condition: (page) => page.fileData.slug !== "index",
    }),
    Component.ArticleTitle(),
    Component.ContentMeta(),
    Component.TagList(),
  ],
  afterBody: [
    Component.ConditionalRender({
      component: Component.Graph({
        scale: 30.0,
        fontSize: 2.0,
        localGraph: {
          removeTags: ["hide"], // Hide nodes with #hide tag
          showTags: true,
        },
        globalGraph: {
          removeTags: ["hide"], // Hide nodes with #hide tag
          showTags: true,
        },
      }),
      // Only show this large graph on the home page
      condition: (page) => page.fileData.slug === "index",
    }),
  ],
  left: [
    Component.PageTitle(),
    Component.MobileOnly(Component.Spacer()),
    Component.Flex({
      components: [
        {
          Component: Component.Search(),
          grow: true,
        },
        { Component: Component.Darkmode() },
        { Component: Component.ReaderMode() },
      ],
    }),
    Component.Explorer({
      filterFn: (node) => {
        // 1. Logic for "index folder specification pages"
        // Hide the file named "index" (the folder note) from the sidebar
        // to avoid duplication with the folder itself.
        if (node.name === "index") return false

        // node.file is undefined for folders, so we keep them visible
        const f = node.file
        if (!f) return true 

        // 2. Logic for "tags: hide"
        // Safely handle tags whether they are an array or a single string
        const tags = f.frontmatter?.tags
        const hasHideTag = Array.isArray(tags) 
          ? tags.includes("hide") 
          : tags === "hide"
          
        // Check for specific 'hide' property (hide: true)
        const hideProp = f.frontmatter?.hide === true

        // If either condition is met, hide the file
        if (hasHideTag || hideProp) {
          return false
        }

        return true
      },
    }),
  ],
  right: [
    Component.Graph({
      localGraph: {
        drag: true,
        zoom: true,
        depth: 1,
        scale: 1.1,
        repelForce: 0.5,
        centerForce: 0.3,
        linkDistance: 30,
        fontSize: 0.6,
        opacityScale: 1,
        showTags: true,
        removeTags: ["hide"],
      },
      globalGraph: {
        drag: true,
        zoom: true,
        depth: -1,
        scale: 0.9,
        repelForce: 0.5,
        centerForce: 0.3,
        linkDistance: 30,
        fontSize: 0.6,
        opacityScale: 1,
        showTags: true,
        removeTags: ["hide"],
      },
    }),
    Component.DesktopOnly(Component.TableOfContents()),
    Component.Backlinks(),
  ],
}

// components for pages that display lists of pages (e.g. tags or folders)
export const defaultListPageLayout: PageLayout = {
  beforeBody: [Component.Breadcrumbs(), Component.ArticleTitle(), Component.ContentMeta()],
  left: [
    Component.PageTitle(),
    Component.MobileOnly(Component.Spacer()),
    Component.Flex({
      components: [
        {
          Component: Component.Search(),
          grow: true,
        },
        { Component: Component.Darkmode() },
      ],
    }),
    Component.Explorer({
      // Apply the same filter logic to the list pages
      filterFn: (node) => {
        if (node.name === "index") return false
        
        const f = node.file
        if (!f) return true 

        const tags = f.frontmatter?.tags
        const hasHideTag = Array.isArray(tags) ? tags.includes("hide") : tags === "hide"
        const hideProp = f.frontmatter?.hide === true

        return !(hasHideTag || hideProp)
      },
    }),
  ],
  right: [],
  afterBody: [],
}