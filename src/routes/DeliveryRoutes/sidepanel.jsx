import {useState} from "react";
import { Link } from "react-router-dom";
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faBox, faHome, faUser, faBoxesPacking, faStreetView, faHeadset, faChainBroken } from '@fortawesome/free-solid-svg-icons'

function Sidepanel()
{

   

    return(

            <div className="sm:w-[300px] w-[80px] side-panel sm:p-5 p-2 flex flex-col items-center bg-[#d7002a] fixed h-screen z-10">

                <div className="sidepalen-body h-2/3 p-3 w-full justify-center">
                    <div className="user-profile sm:flex flex-row items-center space-x-5 hidden">
                        <label className="text-[28px]">
                            <FontAwesomeIcon icon={faUser} color="white" />
                        </label>
                        <span className="user-name text-[22px] text-white">User Name</span>
                    </div>
                    


                    <div className="nav-body w-full flex-col justify-center space-y-5 mt-10">
                            <ul className="space-y-5">
                                <Link to="/Dashboard"><li className="p-2 w-full bg-white rounded-md mb-5">
                                    <FontAwesomeIcon icon={faHome} className="mr-5"/><span className="sm:inline-flex hidden">Dashboard</span></li></Link>
                                <Link to="/CreateDelivery"><li className="p-2 w-full bg-white rounded-md mb-5">
                                    <FontAwesomeIcon icon={faBox} className="mr-5"/><span className="sm:inline-flex hidden">Create Delivery</span></li></Link>

                                <Link to="/Orders"><li className="p-2 w-full bg-white rounded-md mb-5">
                                    <FontAwesomeIcon icon={faBoxesPacking} className="mr-5"/><span className="sm:inline-flex hidden">Orders</span></li></Link>

                                <Link to="/RoutePlanning"><li className="p-2 w-full bg-white rounded-md mb-5">
                                    <FontAwesomeIcon icon={faStreetView} className="mr-5"/><span className="sm:inline-flex hidden">Route Planning</span></li></Link>
                               
                            </ul>
                    </div>


                </div>
               

            </div>
            

    );
}


export default Sidepanel;