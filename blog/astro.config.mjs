import { defineConfig } from 'astro/config';

export default defineConfig({
  integrations: [],
  site: 'https://machina.chat',
  base: '/blog',
  markdown: {
    shikiConfig: {
      theme: 'github-dark',
    },
  },
});
