export interface TestCase {
  query: string
  locale: string
  contentType: string
  description: string
  group: string
}

export const TEST_CASES: TestCase[] = [
  // ── English — semantic ──────────────────────────────────────────────────
  { query: 'making government more efficient',                     locale: 'en',    contentType: '',               description: 'Vague intent, no exact keywords',          group: 'EN — semantic' },
  { query: 'building trust between citizens and institutions',     locale: 'en',    contentType: '',               description: 'Abstract concept',                         group: 'EN — semantic' },
  { query: 'how to lead change in the public sector',             locale: 'en',    contentType: '',               description: 'Conversational / question-style',           group: 'EN — semantic' },
  { query: 'what skills do civil servants need for the future',   locale: 'en',    contentType: '',               description: 'Question-style query',                     group: 'EN — semantic' },
  { query: 'learning and development for government employees',   locale: 'en',    contentType: 'course',         description: 'Semantic + course filter',                 group: 'EN — semantic' },

  // ── English — exact / acronym ───────────────────────────────────────────
  { query: 'GDPR compliance framework',                           locale: 'en',    contentType: '',               description: 'Acronym — tests sparse exact match',       group: 'EN — exact/acronym' },
  { query: 'SDGs sustainable development goals',                  locale: 'en',    contentType: '',               description: 'Acronym + full expansion',                 group: 'EN — exact/acronym' },
  { query: 'net zero carbon emissions policy',                    locale: 'en',    contentType: '',               description: 'Technical policy term',                    group: 'EN — exact/acronym' },
  { query: 'open government data OGD',                            locale: 'en',    contentType: '',               description: 'Mixed acronym and phrase',                 group: 'EN — exact/acronym' },
  { query: 'digital transformation',                              locale: 'en',    contentType: '',               description: 'Classic benchmark',                        group: 'EN — exact/acronym' },

  // ── English — content type filters ─────────────────────────────────────
  { query: 'leadership workshop',                                 locale: 'en',    contentType: 'event',          description: 'Event filter',                             group: 'EN — filtered' },
  { query: 'data analysis skills',                                locale: 'en',    contentType: 'course',         description: 'Course filter',                            group: 'EN — filtered' },
  { query: 'climate adaptation policy',                           locale: 'en',    contentType: 'solutionArticle',description: 'Article filter',                           group: 'EN — filtered' },

  // ── French ──────────────────────────────────────────────────────────────
  { query: 'transformation numérique',                            locale: 'fr',    contentType: '',               description: 'Equiv: "digital transformation"',          group: 'FR' },
  { query: 'données ouvertes gouvernement',                       locale: 'fr',    contentType: '',               description: 'Equiv: "open government data"',            group: 'FR' },
  { query: 'engagement citoyen',                                  locale: 'fr',    contentType: '',               description: 'Equiv: "citizen engagement"',              group: 'FR' },
  { query: 'innovation secteur public',                           locale: 'fr',    contentType: '',               description: 'Equiv: "public sector innovation"',        group: 'FR' },

  // ── French Canadian — limited corpus, tests B fallback ─────────────────
  { query: 'transparence et responsabilité',                      locale: 'fr-CA', contentType: '',               description: 'Low-volume locale, tests B fallback',      group: 'FR-CA' },
  { query: 'gouvernement numérique',                              locale: 'fr-CA', contentType: '',               description: 'Low-volume locale',                        group: 'FR-CA' },

  // ── German — no Snowball stemmer in B's TS port ─────────────────────────
  { query: 'digitale Verwaltung',                                 locale: 'de',    contentType: '',               description: 'Equiv: "digital government"',              group: 'DE' },
  { query: 'öffentlicher Dienst Innovation',                      locale: 'de',    contentType: '',               description: 'Inflected German, tests stemming gap',     group: 'DE' },
  { query: 'Beschaffungsreform',                                  locale: 'de',    contentType: '',               description: 'Single compound German word',              group: 'DE' },

  // ── Spanish ─────────────────────────────────────────────────────────────
  { query: 'transformación digital',                              locale: 'es',    contentType: '',               description: 'Equiv: "digital transformation"',          group: 'ES' },
  { query: 'datos abiertos gobierno',                             locale: 'es',    contentType: '',               description: 'Equiv: "open government data"',            group: 'ES' },
  { query: 'innovación sector público',                           locale: 'es',    contentType: '',               description: 'Equiv: "public sector innovation"',        group: 'ES' },

  // ── Portuguese Brazil ───────────────────────────────────────────────────
  { query: 'transformação digital',                               locale: 'pt-BR', contentType: '',               description: 'Equiv: "digital transformation"',          group: 'PT-BR' },
  { query: 'governo aberto',                                      locale: 'pt-BR', contentType: '',               description: 'Equiv: "open government"',                 group: 'PT-BR' },

  // ── Arabic ──────────────────────────────────────────────────────────────
  { query: 'الحوكمة الرقمية',                                     locale: 'ar',    contentType: '',               description: '"Digital governance"',                     group: 'AR' },
  { query: 'البيانات المفتوحة',                                   locale: 'ar',    contentType: '',               description: '"Open data"',                              group: 'AR' },

  // ── Japanese — whitespace fallback in B ─────────────────────────────────
  { query: 'デジタル変革',                                          locale: 'ja',    contentType: '',               description: '"Digital transformation"',                 group: 'JA' },
  { query: '電子政府サービス',                                      locale: 'ja',    contentType: '',               description: '"E-government services"',                  group: 'JA' },

  // ── Korean ──────────────────────────────────────────────────────────────
  { query: '디지털 전환',                                          locale: 'ko',    contentType: '',               description: '"Digital transformation"',                 group: 'KO' },
  { query: '공공 서비스 혁신',                                      locale: 'ko',    contentType: '',               description: '"Public service innovation"',              group: 'KO' },

  // ── Indonesian ──────────────────────────────────────────────────────────
  { query: 'transformasi digital',                                locale: 'id',    contentType: '',               description: 'Equiv: "digital transformation"',          group: 'ID' },
  { query: 'inovasi pemerintah',                                  locale: 'id',    contentType: '',               description: 'Equiv: "government innovation"',           group: 'ID' },
]

export const GROUPS = [...new Set(TEST_CASES.map((tc) => tc.group))]
