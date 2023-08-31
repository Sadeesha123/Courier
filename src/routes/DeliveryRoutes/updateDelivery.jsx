import {useState, useEffect} from "react";
import { Link,  useNavigate} from "react-router-dom";
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faPlus, faStarHalfStroke } from '@fortawesome/free-solid-svg-icons'
import Sidepanel from "../../components/sidepanel";
import { post } from "../../Api";
import bg from '../../images/mainbg1.jpg';




import {
    BrowserRouter as Router,
    Route,
    useParams
  } from "react-router-dom";

function UpdateDelivery()
{
    const navigate = useNavigate();
    const [branchpickup, setbranchpickup] = useState();
    const [vehicle, setvehicle] = useState();
    const [date, setdate] = useState();
    const [destination, setdestination] = useState();
    const [orderid, setorderid] = useState();
    const [status, setstatus] = useState();
    const [customerid, setcustomerid] = useState();
    const [estimateddate, setestimateddate] = useState();
    const [telephonenumber, settelephonenumber] = useState();
    const [departuredate, setdeparturedate] = useState();

    const { id } = useParams();

    const getData = async () =>{
        try {
            const response = await post('/api/order/info', {id:id}, {});
                if(response){
                    console.log(response)
                    setbranchpickup(response.data.branch_pickup)
                    setvehicle(response.data.vehicle)
                    setdestination(response.data.destination)
                    setdate(response.data.date)
                    setorderid(response.data.order_id)
                    setstatus(response.data.branch_pickup)
                    setcustomerid(response.data.customer_id)
                    setestimateddate(response.data.estimated_date)
                    settelephonenumber(response.data.telephone_number)
                    setdeparturedate(response.data.departure_date)
                }
            
        } catch (error) {
            console.error(error);
        }
    }

    useEffect(() => {
            getData();       
      }, []);


    const handleSubmit = (event) => {
        event.preventDefault();
        
      };


      const handleUpdate = async (event) =>  {
        var postData = {
            id:id,
            branch_pickup:branchpickup,
            vehicle: vehicle,
            date: date,
            destination: destination,
            order_id: orderid,
            status: status,
            estimated_date: estimateddate,
            customer_id: customerid,
            telephone_number: telephonenumber,
            departure_date: departuredate,
            
        };
        console.log(postData)
        try {
            const response = await post('/api/order/update', postData, {});
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
                                Update Delivery
                            </h1>
                            <form onSubmit={handleSubmit} className="flex flex-col w-full">
                                <label htmlFor="customerId" className="mb-2 font-semibold text-gray-600">
                                    Customer ID
                                    </label>
                                    <input value={customerid} onChange={(e)=>setcustomerid(e.target.value)} type="text" id="customerId" name="customerId" className="mb-4 p-2 rounded-lg border border-gray-300" required/>


                                    <label htmlFor="orderId" className="mb-2 font-semibold text-gray-600">
                                    Order ID
                                    </label>
                                    <input value={orderid} onChange={(e)=>setorderid(e.target.value)} type="text" id="orderId" name="orderId" className="mb-4 p-2 rounded-lg border border-gray-300" required/>

                                    <label htmlFor="pickup" className="mb-2 font-semibold text-gray-600">
                                    Pickup
                                    </label>
                                    <input value={branchpickup} onChange={(e)=>setbranchpickup(e.target.value)} type="text" id="pickup" name="pickup" className="mb-4 p-2 rounded-lg border border-gray-300" required/>

                                    <label htmlFor="destination" className="mb-2 font-semibold text-gray-600">
                                    Destination
                                    </label>
                                    <input value={destination} onChange={(e)=>setdestination(e.target.value)} type="text" id="destination" name="destination" className="mb-4 p-2 rounded-lg border border-gray-300" required/>

                                    <label htmlFor="date" className="mb-2 font-semibold text-gray-600">
                                    Estimated Date
                                    </label>
                                    <input value={date} onChange={(e)=>setdate(e.target.value)} type="date" id="date" name="date" className="mb-4 p-2 rounded-lg border border-gray-300" required/>

                                    <label htmlFor="departureDate" className="mb-2 font-semibold text-gray-600">
                                    Departure Date
                                    </label>
                                    <input value={departuredate} onChange={(e)=>setdeparturedate(e.target.value)} type="date" id="departureDate" name="departureDate" className="mb-4 p-2 rounded-lg border border-gray-300" required />


                                    <label htmlFor="tp" className="mb-2 font-semibold text-gray-600">
                                    Telephone number
                                    </label>
                                    <input value={telephonenumber} onChange={(e)=>settelephonenumber(e.target.value)} type="text" id="tp" name="tp" className="mb-4 p-2 rounded-lg border border-gray-300" required/>


                                    <label htmlFor="vehicle" className="mb-2 font-semibold text-gray-600">
                                    Vehicle
                                    </label>
                                    <select value={vehicle} onChange={(e)=>setvehicle(e.target.value)} id="vehicle" name="vehicle" className="mb-4 p-2 rounded-lg border border-gray-300">
                                    <option value="Please Select">Please Select</option>
                                      <option value="car">Car</option>
                                      <option value="bike">Bike</option>
                                      <option value="truck">Truck</option>
                                    </select>

                                    <label htmlFor="status" className="mb-2 font-semibold text-gray-600">
                                    Status
                                    </label>
                                    <select value={status} onChange={(e)=>setstatus(e.target.value)} id="status" name="status" className="mb-4 p-2 rounded-lg border border-gray-300">
                                    <option value="Processing">Processing</option>
                                    <option value="Delivered">Delivered</option>
                                    </select>

                                    <button onClick={handleUpdate} className="py-2 px-4 bg-gradient-to-r from-blue-500 to-blue-700 text-white rounded-lg w-52 ml-auto mt-8">
                                    Update Delivery
                                    </button>
                                </form>





                            </div>


                          </div>
                        </div>
                        
                    </div>
            </div>

    );
}


export default UpdateDelivery;