import "@testing-library/jest-dom/vitest"

// jsdom has no layout, so it has no scrollIntoView; the evidence mark calls it.
Element.prototype.scrollIntoView = () => {}
