// The component's unit tests run under jsdom with the wasm class mocked.
import { defineConfig } from "vitest/config";

export default defineConfig({
    test: { environment: "jsdom", include: ["test/**/*.test.tsx"] },
});
