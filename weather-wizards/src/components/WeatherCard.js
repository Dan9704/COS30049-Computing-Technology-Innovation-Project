import React from 'react'
import {
  FeedSummary,
  FeedLabel,
  FeedEvent,
  FeedDate,
  FeedContent,
  CardHeader,
  CardContent,
  Card,
  Feed,
} from 'semantic-ui-react'

const WeatherCard = () => (
  <Card>
    <CardContent>
      <CardHeader>City</CardHeader>
    </CardContent>
    <CardContent>
      <Feed>
        <FeedEvent>
          <FeedContent>
            <h5>Date</h5>
            <div className='weather-card'>
                <div className='weather-card-child'>
                    Temperature
                </div>
                <div className='weather-card-child'>
                    Humidity
                </div>
                <div className='weather-card-child'>
                    Wind
                </div>
            </div>
          </FeedContent>
        </FeedEvent>

        
      </Feed>
    </CardContent>
  </Card>
)

export default WeatherCard