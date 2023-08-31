import {useEffect, useState} from "react";
import { Link } from "react-router-dom";
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faPlus, faStarHalfStroke } from '@fortawesome/free-solid-svg-icons'
import Sidepanel from "../../components/sidepanel";
import {
    BrowserRouter as Router,
    Route,
    useParams
  } from "react-router-dom";

  import { post } from "../../Api";
function SingleOrder()
{
    const [data, setData] = useState(
        {
            branch_pickup: "",
            customer_id: 1,
            date: "",
            departure_date: "",
            destination: "",
            estimated_date: "",
            id: 1,
            order_id: "",
            package_id: null,
            status: "",
            telephone_number: "",
            vehicle: ""
        }
    )
    const { id } = useParams();

    const getData = async () =>{
        try {
            const response = await post('/api/order/info', {id:id}, {});
                if(response){
                    console.log(response)
                    setData(response.data)
                }
            
        } catch (error) {
            console.error(error);
        }
    }

    useEffect(() => {
            getData();       
      }, []);

   

    return(

            <div className="main-body h-screen w-full bg-slate-100">
                    <div className="main-body-container h-screen w-full flex flex-row">
                        <Sidepanel/>
                        <div className="w-5/6 side-panel bg-slate-100 p-5">
                          <div className="common-body p-5 flex flex-col h-full items-center justify-center">

                            <div className="form-body w-[60%] flex flex-col p-5 mx-auto  justify-left bg-white rounded-lg shadow-md shadow-slate-300">
                                
                            

                             <div className="order-details flex flex-col space-y-[20px]">
                                <span>Customer Id : #{data.customer_id}</span>
                                <span>Order Id : #{data.order_id}</span>
                                <span>Pick Up : {data.branch_pickup}</span>
                                <span>Destination : {data.destination}</span>
                                <span>Date : {data.date}</span>
                                <span>Contact : {data.telephone_number}</span>
                             </div>



                            

                            

                            </div>


                          </div>
                        </div>
                        
                    </div>
            </div>

    );
}


export default SingleOrder;