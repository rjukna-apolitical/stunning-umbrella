import { Body, Controller, Get, Post, Query } from '@nestjs/common';
import { SearchRequest, SearchService } from './search.service';

@Controller('search')
export class SearchController {
  constructor(private readonly searchService: SearchService) {}

  /**
   * POST /search
   * Full search with all options.
   */
  @Post()
  search(@Body() body: SearchRequest) {
    return this.searchService.search(body);
  }

  /**
   * GET /search?q=...&locale=en
   * Convenience endpoint for quick testing from the browser or curl.
   */
  @Get()
  searchGet(
    @Query('q') query: string,
    @Query('locale') locale = 'en',
    @Query('contentType') contentType?: string,
    @Query('page') page?: string,
    @Query('pageSize') pageSize?: string,
    @Query('crossLingual') crossLingual?: string,
  ) {
    return this.searchService.search({
      query,
      locale,
      contentType,
      page: page ? parseInt(page, 10) : 1,
      pageSize: pageSize ? parseInt(pageSize, 10) : 10,
      crossLingual: crossLingual === 'true',
    });
  }
}
