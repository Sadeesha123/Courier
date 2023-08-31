import { useState } from "react";
import { Link } from "react-router-dom";
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faPlus, faStarHalfStroke } from '@fortawesome/free-solid-svg-icons'
import Sidepanel from "../../components/sidepanel";
import {MessageDialog} from "../../components/MessageDialog";






function MoreInfo() {




    const handleSubmit = (event) => {
        event.preventDefault();

    };




    return (

        <div className="main-body h-screen w-full bg-slate-100">
            <div className="main-body-container w-full flex flex-row absolute">
                <Sidepanel />
            </div>
            {/* <MessageDialog click={true}/> */}
                {/* Card UI */}
                <div className="w-5/6 side-panel p-5 md:ml-[300px] ml-16">
                <div class=" w-full h-full grid mb-8 border border-gray-200 rounded-lg shadow-sm dark:border-gray-700 md:mb-12 md:grid-cols-2">
                  
                  <div className="border ">
                    <div class="flex flex-col items-center justify-center px-8 py-8 text-center bg-white border-b border-gray-200 rounded-t-lg md:rounded-t-none md:rounded-tl-lg md:border-r   w-full h-full">
                        
                        <div className="max-w-sm p-6 w-full h-full bg-white border border-gray-200 rounded-lg shadow bg-[#E5E4E2] dark:border-gray-700">

                            <br/>

                            <h1 class="text-lg font-semibold text-gray-900 first-line:text-black">Reciever Information</h1> <br/>
                            <br/><br/>

                            <label htmlFor="recieverId" className="mb-2 font-semibold text-gray-600">
                            Reciever ID 
                            </label>
                            &nbsp;  &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; <input className=" rounded-sm border border-gray-300" type="text" placeholder=" Backend DATA" readonly="readonly"/> 
                            <br/>  <br/> <br/>
                            <label htmlFor="email" className="mb-2 font-semibold text-gray-600">
                            Email
                            </label>
                            &nbsp;  &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<input type="text" className=" rounded-sm border border-gray-300" placeholder="Backend DATA" readonly="readonly"/> 
                            <br/>  <br/> <br/>
                            <label htmlFor="mobileNumber" className="mb-2 font-semibold text-gray-600">
                            Mobile Number
                            </label>
                            &nbsp;  &nbsp;<input type="text" className=" rounded-sm border border-gray-300" placeholder="Backend DATA" readonly="readonly"/> 
                            </div>
                            
                            
                        
                        
                        </div> 

                    </div>

                    <div className="border ">
                    <div class="flex flex-col items-center h-full justify-center p-8 text-center bg-white border-b border-gray-200 rounded-tr-lg">
                    
                    <div className="max-w-sm p-6 w-full h-full bg-white border border-gray-200 rounded-lg shadow bg-[#E5E4E2] dark:border-gray-700">
                    <br/><br/>
                            <h1 class="text-lg font-semibold text-gray-900 text-dark">Traffic Details</h1> <br/>
                            <br/><br/>

                            <label htmlFor="trafficCondition" className="mb-2 font-semibold text-gray-600">
                            Traffic Condition 
                            </label>
                            &nbsp; <input type="text" className=" rounded-sm border border-gray-300" placeholder=" Backend DATA" readonly="readonly"/> 
                        <br/>  <br/> <br/>
                            <label htmlFor="skill" className="mb-2 font-semibold text-gray-600">
                            Driver Skill
                            </label>
                            &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<input type="text" className=" rounded-sm border border-gray-300" placeholder=" Backend DATA" readonly="readonly"/> 
                            <br/>  <br/>
                        
    
                          </div> 
                    </div>

                    </div>
                    <div className="border ">
                    <div class="flex flex-col items-center h-full justify-center p-8 text-center bg-white border-b border-gray-200 rounded-tr-lg ">
                    
                    <div className="max-w-sm p-6 w-full h-full bg-white border border-gray-200 rounded-lg shadow bg-[#E5E4E2] dark:border-gray-700">
                    <br/><br/>
                            <h1 class="text-lg font-semibold text-gray-900 text-dark">Prediction Details</h1> <br/>
                            <br/><br/>

                            <label htmlFor="deliverystat" className="mb-2 font-semibold text-gray-600">
                           Delivery Success
                            </label>
                            &nbsp;  &nbsp; &nbsp;<input type="text"  className=" rounded-sm border border-gray-300" placeholder=" Backend DATA" readonly="readonly"/> 
                        <br/>  <br/> <br/>
                             
                          </div> 
                    </div>

                    </div>

                    <div className="border  ">
                    <div class="flex flex-col items-center h-full justify-center p-8 text-center bg-white border-b border-gray-200 rounded-tr-lg ">
                    
                    <div className="max-w-sm p-6 w-full h-full bg-white border border-gray-200 rounded-lg shadow bg-[#E5E4E2] dark:border-gray-700">
                    <br/><br/>
                            <h1 class="text-lg font-semibold text-gray-900 text-dark">Sender Details</h1> <br/>
                            <br/><br/>

                            <label htmlFor="senderMobile" className="mb-2 font-semibold text-gray-600">
                            Sender Mobile
                            </label>
                            &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp;<input type="text" className=" rounded-sm border border-gray-300" placeholder=" Backend DATA" readonly="readonly"/> 
                        <br/>  <br/> <br/>
                            <label htmlFor="orderId" className="mb-2 font-semibold text-gray-600">
                            Number of Pieces                            </label>
                            &nbsp; <input type="text" placeholder=" Backend DATA" className=" rounded-sm border border-gray-300" readonly="readonly"/> 
                            <br/>  <br/>
                        
    
                          </div> 
                    </div>

                    </div>

                </div>
                
        </div>
        </div>


    );
}


export default MoreInfo;


