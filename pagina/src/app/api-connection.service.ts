import { Injectable } from '@angular/core';
import { HttpClient, HttpClientModule } from '@angular/common/http';

@Injectable({
  providedIn: 'root'
})
export class ApiConnectionService {

  constructor(private http: HttpClient) { }

  getGraph(postData){
    return this.http.post("http://127.0.0.1:5000/callMethod",postData)
  }
}
