import { useState } from "react";
import { Link } from "react-router-dom";
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faBox, faUser, faBoxesPacking, faStreetView, faHome, faHeadset, faChainBroken, faTable } from '@fortawesome/free-solid-svg-icons'

function Sidepanel2() {



    return (

        <div className="w-[300px] side-panel p-5 flex flex-col  items-center bg-[conic-gradient(at_bottom_left,_var(--tw-gradient-stops))] bg-[#d7002a]">

            <div className="justify-center w-full p-5 sidepalen-body h-2/3">
                <div className="flex flex-row items-center space-x-5 user-profile">
                    <label className="text-[28px]">
                        <FontAwesomeIcon icon={faUser} color="white" />
                    </label>
                    <span className="user-name text-[22px] text-white">User Name</span>
                </div>



                <div className="flex-col justify-center w-full mt-10 space-y-5 nav-body">
                    <ul className="space-y-5">
                        <Link to="/Dashboard"><li className="w-full p-2 mb-5 rounded-md bg-white">
                            <FontAwesomeIcon icon={faHome} className="mr-5" />Dashboard</li></Link>
                        <Link to="/PerformanceTrack"><li className="w-full p-2 mb-5 rounded-md bg-white">
                            <FontAwesomeIcon icon={faTable} className="mr-5" />Performance Track</li></Link>
                        <Link to="/CreateOrder"><li className="w-full p-2 mb-5 rounded-md bg-white">
                            <FontAwesomeIcon icon={faBox} className="mr-5" />Create Order</li></Link>

                        <Link to="/CustomerChurn"><li className="w-full p-2 mb-5 rounded-md bg-white">
                            <FontAwesomeIcon icon={faBoxesPacking} className="mr-5" />Customer Charn</li></Link>

                        <Link to="/ChurnRank"><li className="w-full p-2 mb-5 rounded-md bg-white">
                            <FontAwesomeIcon icon={faBoxesPacking} className="mr-5" />Churn Rank</li></Link>

                        <Link to="/DeliveryRank"><li className="w-full p-2 mb-5 rounded-md bg-white">
                            <FontAwesomeIcon icon={faStreetView} className="mr-5" />Delivery Success Rank</li></Link>

                    </ul>
                </div>


            </div>


        </div>


    );
}


export default Sidepanel2;