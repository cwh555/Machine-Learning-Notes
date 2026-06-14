import { PageLayout, SharedLayout } from "./quartz/cfg"
import * as Component from "./quartz/components"

// 1. SHARED COMPONENTS (Header, Footer, and the Index Graph)
export const sharedPageComponents: SharedLayout = {
  head: Component.Head(),
  header: [],
  afterBody: [
    Component.PrivateBackground(),
    Component.PrivateAnimation(),
    // This graph only renders on the homepage (index)
    Component.ConditionalRender({
      component: Component.Graph({
        scale: 30.0,
        fontSize: 2.0,
        localGraph: {
          removeTags: ["hide"],
          showTags: true,
        },
        globalGraph: {
          removeTags: ["hide"],
          showTags: true,
        },
      }),
      condition: (page) => page.fileData.slug === "index",
    }),
  ],
  footer: Component.Footer({
    links: {
      github: { text: "GitHub", url: "https://github.com/cwh555/Machine-Learning-Notes" }
    },
    text: {
      contact: { text: "Contact : bergerchen719@gmail.com" }
    }
  }),
}

// 2. CONTENT PAGES (Single Notes)
export const defaultContentPageLayout: PageLayout = {
  beforeBody: [
    Component.ConditionalRender({
      component: Component.Breadcrumbs(),
      condition: (page) => page.fileData.slug !== "index",
    }),
    Component.ArticleTitle(),
    Component.ContentMeta(),
    Component.TagList(),
    Component.ConditionalRender({
      component: Component.PrivateBackgroundSettings(),
      condition: (page) => page.fileData.slug === "action-settings",
    }),
    Component.ConditionalRender({
      component: Component.PrivateAnimationSettings(),
      condition: (page) => page.fileData.slug === "action-settings",
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
        { Component: Component.ProtectedGate() },
      ],
    }),
    Component.Explorer({
      filterFn: (node) => {
        if (node.slugSegment === "tags") return false
        if (node.data?.properties?.includes("hide")) return false
        return true
      }
    })
  ],
  right: [
    // Component.Graph({
    //   localGraph: {
    //     drag: true,
    //     zoom: true,
    //     depth: 1,
    //     scale: 1.1,
    //     repelForce: 0.5,
    //     centerForce: 0.3,
    //     linkDistance: 30,
    //     fontSize: 0.6,
    //     opacityScale: 1,
    //     showTags: true,
    //     removeTags: ["hide"],
    //   },
    //   globalGraph: {
    //     drag: true,
    //     zoom: true,
    //     depth: -1,
    //     scale: 0.9,
    //     repelForce: 0.5,
    //     centerForce: 0.3,
    //     linkDistance: 30,
    //     fontSize: 0.6,
    //     opacityScale: 1,
    //     showTags: true,
    //     removeTags: ["hide"],
    //   },
    // }),
    Component.DesktopOnly(Component.PageImage()),
    Component.DesktopOnly(Component.TableOfContents()),
  ],
  // Standard body for content pages
  pageBody: Component.Content(), 
}

// 3. LIST PAGES (Folders/Tags)
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
      filterFn: (node) => {
        if (node.slugSegment === "tags") return false
        if (node.data?.properties?.includes("hide")) return false
        return true
      }
    }),
  ],
  right: [
    // Component.Graph({
    //   localGraph: {
    //     drag: true,
    //     zoom: true,
    //     depth: 1,
    //     scale: 1.1,
    //     repelForce: 0.5,
    //     centerForce: 0.3,
    //     linkDistance: 30,
    //     fontSize: 0.6,
    //     opacityScale: 1,
    //     showTags: true,
    //     removeTags: ["hide"],
    //   },
    //   globalGraph: {
    //     drag: true,
    //     zoom: true,
    //     depth: -1,
    //     scale: 0.9,
    //     repelForce: 0.5,
    //     centerForce: 0.3,
    //     linkDistance: 30,
    //     fontSize: 0.6,
    //     opacityScale: 1,
    //     showTags: true,
    //     removeTags: ["hide"],
    //   },
    // }),
    Component.DesktopOnly(Component.TableOfContents()),
  ],
}