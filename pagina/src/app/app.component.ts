import { Component } from '@angular/core';
import {ApiConnectionService} from './api-connection.service'

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  constructor(private apiService:ApiConnectionService) { }
  title = 'paginaWeb';
  ruta=""
  graphBase64:any
  graph(){
    console.log(this.ruta)
    this.apiService.getGraph({"link":this.ruta}).subscribe(x=>{
      this.graphBase64=x
      console.log(this.graphBase64)
    }

    )
  }
}
