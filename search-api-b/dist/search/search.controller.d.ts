import { SearchRequest, SearchService } from './search.service';
export declare class SearchController {
    private readonly searchService;
    constructor(searchService: SearchService);
    search(body: SearchRequest): Promise<import("./search.service").SearchResponse>;
    searchGet(query: string, locale?: string, contentType?: string, page?: string, pageSize?: string, crossLingual?: string): Promise<import("./search.service").SearchResponse>;
}
