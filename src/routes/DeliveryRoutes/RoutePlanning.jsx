import {useState} from "react";
import { Link } from "react-router-dom";
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faPlus, faStarHalfStroke } from '@fortawesome/free-solid-svg-icons'
import Sidepanel from "../../components/sidepanel";

import bg2 from '../../images/bg2.jpg';


function RoutePlanning()
{
  const [pickup, setPickup] = useState("");
    const [delivery, setDelivery] = useState("");
    const [mapUrl, setMapUrl] = useState("");

    const handleSubmit = async (event) => {
        event.preventDefault();

        try {
            const response = await fetch('/maps', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    pickup: pickup,
                    delivery: delivery
                })
            });

            if (response.ok) {
                const data = await response.json();
                if (data.status === "success") {
                    setMapUrl(data.map);
                }
            } else {
                console.error('Error fetching route data');
            }
        } catch (error) {
            console.error('Error fetching route data:', error);
        }
    };


    // const handleSubmit = (event) => {
    //     event.preventDefault();
        
    //   };


   

    return(

            <div className="main-body h-screen w-full bg-slate-100">
             <img src={bg2} alt="" srcset="" className="object-cover w-[100%] h-[100%]" />
                    <div className="main-body-container h-screen w-full flex flex-row absolute top-0 bg-black bg-opacity-50">
                        <Sidepanel/>
                        <div className="w-5/6 side-panel  p-5 sm:ml-[300px] ml-16">
                          <div className="common-body p-5 flex flex-col h-full items-center justify-center">

                            <div className="form-body w-[90%] flex md:flex-row flex-col p-5 mx-auto items-center justify-center bg-white rounded-lg shadow-md shadow-slate-300">
                                
                             <div className="map sm:w-[60%] w-full text-center p-2 h-full">
                              
                             <h1 className="flex items-center justify-center pt-4 text-xl uppercase font-bold pb-4">
                                Route 
                            </h1>
                                {/* <span>G-MAP will appear here</span> */}
                                <div className="p-2 h-[200px] sm:h-[400px] bg-slate-400 mr-8">
                                {mapUrl && (
                                <iframe
                                    width="100%"
                                    height="100%"
                                    frameborder="0"
                                    marginheight="0"
                                    marginwidth="0"
                                    title="map"
                                    scrolling="no"
                                    src={mapUrl}
                                ></iframe>
                            )}
                                {/* <iframe width="100%" height="100%" frameborder="0" marginheight="0" marginwidth="0" title="map" scrolling="no" src="https://maps.google.com/maps?width=100%&height=600&hl=en&q=%C4%B0zmir+(My%20Business%20Name)&ie=UTF8&t=&z=14&iwloc=B&output=embed" ></iframe> */}
                                </div>
                             </div>

                             <div className="order-details flex flex-col space-y-[20px] ">
                             <form onSubmit={handleSubmit} className="flex flex-col w-full ">
                                <label htmlFor="orderId" className="mb-2 font-semibold text-gray-600 ">
                                Order ID
                                </label>
                                <input type="text" id="orderId" name="orderId" className="mb-4 p-2 rounded-lg border border-gray-300" required/>
                                <label htmlFor="pickUp" className="mb-2 font-semibold text-gray-600">
                                Pickup (latitude, longitude)
                                </label>
                                <input type="text" id="pickup" name="pickup" value={pickup} onChange={(e) => setPickup(e.target.value)} className="mb-4 p-2 rounded-lg border border-gray-300" required/>
                                <label htmlFor="twoMoreSteps" className="mb-2 font-semibold text-gray-600">
                                Two more Steps
                                </label>
                                <input type="text" id="twoMoreSteps" name="twoMoreSteps" className="mb-4 p-2 rounded-lg border border-gray-300" required />
                                <label htmlFor="delivery" className="mb-2 font-semibold text-gray-600">
                                Delivery (latitude, longitude)
                                </label>
                                <input type="text" id="delivery" name="delivery" value={delivery} onChange={(e) => setDelivery(e.target.value)} className="mb-4 p-2 rounded-lg border border-gray-300" required/>
                                <button onClick={handleSubmit} className="py-2 px-4 bg-gradient-to-r from-red-500 to-red-700 text-white rounded-lg w-52 ml-auto mt-8 ml-4">
                                    Route
                                  </button>
                                {/* <span>Order Id : #TJH097</span>
                                <span>Pick Up</span>
                                <span>Two more Steps</span>
                                <span>Delivery</span> */}
                              </form>
                             </div>



                            </div>


                          </div>
                        </div>
                        
                    </div>
            </div>

    );
}


export default RoutePlanning;