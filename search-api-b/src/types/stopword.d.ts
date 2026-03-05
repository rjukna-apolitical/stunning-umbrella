declare module 'stopword' {
  export const eng: string[];
  export const fra: string[];
  export const deu: string[];
  export const spa: string[];
  export const por: string[];
  export const ita: string[];
  export const ind: string[];
  export const arb: string[];
  export const pol: string[];
  export function removeStopwords(tokens: string[], stopwords: string[]): string[];
}
