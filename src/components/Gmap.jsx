import React, { useEffect, useState } from 'react';
import { GoogleMap, Marker, InfoWindow } from '@react-google-maps/api';

const Gmap = () => {
  const [map, setMap] = useState(null);
  const [userLocation, setUserLocation] = useState(null);

  // Function to get the user's current location
  const getUserLocation = () => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          setUserLocation({
            lat: position.coords.latitude,
            lng: position.coords.longitude,
          });
        },
        (error) => {
          console.error(error);
        }
      );
    } else {
      console.error('Geolocation is not supported by this browser.');
    }
  };

  // Function to handle when the map is loaded
  const handleMapLoad = (map) => {
    setMap(map);
  };

  useEffect(() => {
    getUserLocation();
  }, []);

  return (
    <div className="map w-[60%]">
      <GoogleMap
        onLoad={handleMapLoad}
        center={userLocation}
        zoom={15}
        mapContainerStyle={{ width: '100%', height: '400px' }}
      >
        {userLocation && <Marker position={userLocation} />}
      </GoogleMap>
    </div>
  );
};

export default Gmap;
