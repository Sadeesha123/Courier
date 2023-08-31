import {useEffect, useState} from "react";
import { Link ,useNavigate} from "react-router-dom";
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faPlus, faStarHalfStroke } from '@fortawesome/free-solid-svg-icons'
import Sidepanel from "../../components/sidepanel";
import { post } from "../../Api";
import bg from '../../images/mainbg1.jpg';



function CreateDelivery()
{



    const navigate = useNavigate();
    const handleSubmit = async (event) =>  {
        event.preventDefault();
        const data = new FormData(event.currentTarget);
        var postData = {
            branch_pickup:data.get('pickup'),
            vehicle: data.get('vehicle'),
            date: data.get('date'),
            destination: data.get('destination'),
            order_id: data.get('orderId'),
            status: 'Processing',
            estimated_date: data.get('date'),
            customer_id: data.get('customerId'),
            telephone_number: data.get('tp'),
            departure_date: data.get('departureDate'),
            
        };
        console.log(postData)
        try {
            const response = await post('/api/order/create', postData, {});
                if(response){
                    console.log(response);

                    navigate('/Orders');
                }
            
        } catch (error) {
            console.error(error);
        }
    };

    
   

    return(

            <div className="main-body h-screen w-full bg-slate-100">
                <img src={bg} alt="" srcset="" className="object-cover w-[100%] h-[100%] fixed" />
                    <div className="main-body-container w-full flex flex-row absolute">
                        <Sidepanel/>
                        <div className="w-5/6 side-panel p-5 md:ml-[300px] ml-16">
                          <div className="common-body p-5 flex flex-col h-full items-center ">

                            <div className="form-body md:w-[80%] w-full flex flex-col p-5 mx-auto items-center justify-center bg-white rounded-lg ">
                            <h1 className="flex items-center justify-center pt-4 text-xl uppercase font-bold pb-4">
                                Create Delivery
                            </h1>
                            <form onSubmit={handleSubmit} className="flex flex-col w-full">
                                <label htmlFor="customerId" className="mb-2 font-semibold text-gray-600">
                                    Customer ID
                                    </label>
                                    <input type="text" id="customerId" name="customerId" className="mb-4 p-2 rounded-lg border border-gray-300" required/>


                                    <label htmlFor="orderId" className="mb-2 font-semibold text-gray-600">
                                    Order ID
                                    </label>
                                    <input type="text" id="orderId" name="orderId" className="mb-4 p-2 rounded-lg border border-gray-300" required />

                                    <label htmlFor="pickup" className="mb-2 font-semibold text-gray-600">
                                    Pickup
                                    </label>
                                    <input type="text" id="pickup" name="pickup" className="mb-4 p-2 rounded-lg border border-gray-300" required />

                                    <label htmlFor="destination" className="mb-2 font-semibold text-gray-600">
                                    Destination
                                    </label>
                                    <input type="text" id="destination" name="destination" className="mb-4 p-2 rounded-lg border border-gray-300" required/>

                                    <label htmlFor="date" className="mb-2 font-semibold text-gray-600">
                                    Estimated Date
                                    </label>
                                    <input type="date" id="date" name="date" className="mb-4 p-2 rounded-lg border border-gray-300" required/>

                                    <label htmlFor="departureDate" className="mb-2 font-semibold text-gray-600">
                                    Departure Date
                                    </label>
                                    <input type="date" id="departureDate" name="departureDate" className="mb-4 p-2 rounded-lg border border-gray-300" required/>


                                    <label htmlFor="tp" className="mb-2 font-semibold text-gray-600">
                                    Telephone number
                                    </label>
                                    <input type="text" id="tp" name="tp" className="mb-4 p-2 rounded-lg border border-gray-300" required/>


                                    <label htmlFor="vehicle" className="mb-2 font-semibold text-gray-600">
                                    Vehicle
                                    </label>
                                    <select id="vehicle" name="vehicle" className="mb-4 p-2 rounded-lg border border-gray-300">
                                    <option value="Please Select">Please Select</option>
                                      <option value="car">Car</option>
                                      <option value="bike">Bike</option>
                                      <option value="truck">Truck</option>
                                    </select>

                                    <button type="submit" className="py-2 px-4 bg-gradient-to-r from-blue-500 to-blue-700 text-white rounded-lg w-52 ml-auto mt-8">
                                    Create Delivery
                                    </button>
                                </form>





                            </div>


                          </div>
                        </div>
                        
                    </div>
            </div>

    );
}


export default CreateDelivery;