import { useEffect } from 'react'; 
import './App.css';
import Header from './components/Header';
import WeatherCard from './components/WeatherCard';

const URL = 'https://api.openweathermap.org/data/2.5/weather'
const API_KEY = ''


function App() {




  useEffect(() => {
    fetch(`${URL}?q=London&appid=${API_KEY}`)
    .then(res => res.json())
    .then(data => console.log(data))
  }, [])

  
  return (
    <div className='main'>
      <Header/>
      <WeatherCard/>
    </div>  
  );
}

export default App;
